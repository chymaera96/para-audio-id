from __future__ import annotations

import json
import math
from pathlib import Path
import time

import torch

from ..audio import BadFileRegistry, load_audio
from ..catalogue import load_catalogue
from .checkpoint import load_audio_lm
from .dataset import collate_causal_documents
from .generation import beam_generate, greedy_generate, prompt_from_audio_tokens
from .losses import causal_audio_id_losses
from .tokenizer import MuQRVQTokenizer


def _checkpoint_tokenizer(checkpoint: dict, device: str) -> MuQRVQTokenizer:
    spec = checkpoint["tokenizer_spec"]
    tokenizer = MuQRVQTokenizer(
        spec["model_name"],
        revision=spec["revision"],
        selected_codebooks=int(spec["selected_codebooks"]),
        sample_rate=int(spec["sample_rate"]),
        device=device,
    )
    if tokenizer.spec.fingerprint != checkpoint["tokenizer_fingerprint"]:
        raise ValueError("Loaded MuQ tokenizer does not match the checkpoint")
    return tokenizer


def evaluate(
    checkpoint_path: str | Path,
    *,
    output: str | Path,
    max_tracks: int | None = None,
    device: str = "cuda",
    beam_width: int = 10,
) -> dict:
    model, vocabulary, cfg, checkpoint = load_audio_lm(checkpoint_path, device)
    tokenizer = _checkpoint_tokenizer(checkpoint, device)
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    records = load_catalogue(cfg["data"]["catalogue"])
    by_track_id = {record.track_id: record for record in records}
    selected_track_ids = list(checkpoint["validation_probe"])
    if max_tracks is not None:
        selected_track_ids = selected_track_ids[:max_tracks]
    positions = [float(value) for value in cfg["evaluation"]["shifted_starts"]]
    bad = BadFileRegistry(cfg["data"]["runtime_bad_files"])
    root = Path(cfg["data"]["audio_root"])
    duration = float(cfg["data"]["segment_duration"])
    targets = []
    greedy_codes = []
    greedy_protocol_valid = []
    rankings = []
    combined_losses = []
    audio_losses = []
    id_losses = []
    boundary_eos_losses = []
    digit_correct = 0.0
    exact_correct = 0.0
    latency = 0.0
    rows = []
    for track_id in selected_track_ids:
        if track_id not in by_track_id:
            raise ValueError(f"Validation probe track {track_id} is missing from the catalogue")
        record = by_track_id[track_id]
        if bad.contains(record.path):
            continue
        for start in positions:
            try:
                waveform = load_audio(
                    root / record.path,
                    sample_rate=tokenizer.sample_rate,
                    start=start,
                    duration=duration,
                    pad=True,
                )
                audio_tokens = tokenizer.tokenize(
                    torch.from_numpy(waveform).unsqueeze(0)
                )[0].cpu()
            except Exception as exc:
                bad.add(record.path, exc)
                break
            example = {
                "audio_tokens": audio_tokens,
                "code": record.code,
                "track_id": record.track_id,
                "document_index": -1,
            }
            batch = collate_causal_documents(
                [example],
                vocabulary,
                int(cfg["model"]["max_position_embeddings"]),
            )
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            started = time.perf_counter()
            with torch.inference_mode():
                logits = model(input_ids, attention_mask)
                losses = causal_audio_id_losses(
                    logits,
                    input_ids,
                    batch["audio_target_mask"].to(device),
                    batch["id_target_mask"].to(device),
                    batch["boundary_target_mask"].to(device),
                    id_digit_weight=float(cfg["train"]["id_digit_weight"]),
                )
                prompt = prompt_from_audio_tokens(audio_tokens.to(device), vocabulary)
                greedy = greedy_generate(model, prompt, vocabulary)
                beam = beam_generate(model, prompt, vocabulary, width=beam_width)
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            latency += time.perf_counter() - started
            targets.append(record.code)
            greedy_codes.append(greedy.code)
            greedy_protocol_valid.append(greedy.ended_with_eos)
            rankings.append(beam)
            combined_losses.append(float(losses["loss"]))
            audio_losses.append(float(losses["audio_loss"]))
            id_losses.append(float(losses["id_loss"]))
            boundary_eos_losses.append(float(losses["boundary_eos_loss"]))
            digit_correct += float(losses["teacher_forced_digit_accuracy"])
            exact_correct += float(losses["teacher_forced_exact_accuracy"])
            rows.append(
                {
                    "track_id": record.track_id,
                    "code": record.code,
                    "path": record.path,
                    "start": start,
                    "greedy": greedy.code,
                    "greedy_ended_with_eos": greedy.ended_with_eos,
                    "beam": [
                        {
                            "code": result.code,
                            "log_probability": result.log_probability,
                            "ended_with_eos": result.ended_with_eos,
                        }
                        for result in beam
                    ],
                }
            )
    if not targets:
        raise RuntimeError("No evaluation queries were tokenized")
    count = len(targets)
    mean_audio_loss = sum(audio_losses) / count
    metrics = {
        "queries": count,
        "loss": sum(combined_losses) / count,
        "audio_loss": mean_audio_loss,
        "audio_perplexity": math.exp(min(20.0, mean_audio_loss)),
        "id_loss": sum(id_losses) / count,
        "boundary_eos_loss": sum(boundary_eos_losses) / count,
        "teacher_forced_digit_accuracy": digit_correct / count,
        "teacher_forced_exact_accuracy": exact_correct / count,
        "greedy_top1": sum(
            target == prediction
            for target, prediction in zip(targets, greedy_codes, strict=True)
        )
        / count,
        "invalid_code_rate": sum(not valid for valid in greedy_protocol_valid) / count,
        "mean_latency_seconds": latency / count,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "peak_inference_memory_bytes": (
            int(torch.cuda.max_memory_allocated())
            if str(device).startswith("cuda")
            else 0
        ),
    }
    reciprocal_rank = 0.0
    for width in (1, 5, 10):
        hits = 0
        for target, ranking in zip(targets, rankings, strict=True):
            codes = [result.code for result in ranking]
            hits += int(target in codes[:width])
            if width == 10 and target in codes:
                reciprocal_rank += 1 / (codes.index(target) + 1)
        metrics[f"beam_top{width}"] = hits / count
    metrics["beam_mrr"] = reciprocal_rank / count
    metrics["external_artifacts"] = [
        "audio_lm_checkpoint",
        "frozen_muq_tokenizer_weights",
        "query_audio",
    ]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"metrics": metrics, "queries": rows}, indent=2) + "\n")
    return metrics
