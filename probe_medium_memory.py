from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import time

import torch
from torch.nn.parallel import DistributedDataParallel

from para_audio_id.audio_lm.dataset import collate_causal_documents
from para_audio_id.audio_lm.losses import (
    degraded_causal_base_losses,
    identifier_logit_distillation_loss,
)
from para_audio_id.audio_lm.model import AudioCausalLM
from para_audio_id.audio_lm.profiles import decoder_profile
from para_audio_id.audio_lm.tokenizer import MuQRVQTokenizer
from para_audio_id.config import load_config


DEFAULT_CANDIDATES = (10, 12, 16, 20, 24, 28, 32, 36, 40)
DOCUMENTS_PER_TRACK = 2
WORLD_SIZE = 4
WARMUP_UPDATES = 2
TIMED_UPDATES = 3
MINIMUM_HEADROOM_PERCENT = 10.0
TARGET_TRACK_SELECTIONS = 72_000_000


def _is_oom(error: BaseException) -> bool:
    return isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(
        error
    ).lower()


def _model_config(config: dict) -> dict:
    result = dict(config)
    result["model"] = dict(config["model"])
    result["model"].update(decoder_profile("medium"))
    result["model"].pop("name", None)
    return result


def validate_candidates(values: list[int]) -> list[int]:
    candidates = list(dict.fromkeys(values))
    if candidates != values:
        raise ValueError("--tracks must not contain duplicates")
    if candidates != sorted(candidates) or any(value <= 0 for value in candidates):
        raise ValueError("--tracks must contain ascending positive integers")
    return candidates


def recommend_candidate(
    results: list[dict], *, minimum_headroom_percent: float
) -> dict:
    eligible = [
        row
        for row in results
        if row.get("success")
        and float(row["peak_headroom_percent"]) >= minimum_headroom_percent
    ]
    if not eligible:
        raise RuntimeError(
            "No successful candidate retained the required "
            f"{minimum_headroom_percent:g}% peak memory headroom"
        )
    best = max(
        eligible,
        key=lambda row: (
            float(row["documents_per_second"]),
            int(row["tracks_per_gpu"]),
        ),
    )
    tracks = int(best["tracks_per_gpu"])
    global_tracks = WORLD_SIZE * tracks
    return {
        "selected_tracks_per_gpu": tracks,
        "documents_per_gpu": tracks * DOCUMENTS_PER_TRACK,
        "world_size": WORLD_SIZE,
        "accumulate_grad_batches": 1,
        "global_tracks_per_optimizer_step": global_tracks,
        "global_documents_per_optimizer_step": (
            global_tracks * DOCUMENTS_PER_TRACK
        ),
        "target_track_selections": TARGET_TRACK_SELECTIONS,
        "resolved_max_steps": math.ceil(TARGET_TRACK_SELECTIONS / global_tracks),
        "minimum_peak_headroom_percent": minimum_headroom_percent,
        "selection_rule": "highest measured documents_per_second among safe candidates",
    }


def _documents(audio_tokens: torch.Tensor, vocabulary) -> dict:
    examples = []
    for row, tokens in enumerate(audio_tokens):
        track = row // DOCUMENTS_PER_TRACK
        examples.append(
            {
                "audio_tokens": tokens,
                "code": f"{track % 100_000:05d}",
                "track_id": f"probe-{track}",
                "document_index": row,
                "segment_duration": 2.0,
                "view_type": "probe",
            }
        )
    return collate_causal_documents(examples, vocabulary, max_positions=512)


def _optimizer_update(
    *,
    model,
    optimizer,
    tokenizer: MuQRVQTokenizer,
    waveforms: torch.Tensor,
    device: torch.device,
) -> None:
    audio_tokens = tokenizer.tokenize(waveforms)
    documents = int(waveforms.shape[0])
    if audio_tokens.shape != (documents, 400):
        raise RuntimeError(
            f"MuQ returned {tuple(audio_tokens.shape)}, expected ({documents}, 400)"
        )
    prepared = _documents(audio_tokens, tokenizer.vocabulary)
    input_ids = prepared["input_ids"].to(device)
    attention_mask = prepared["attention_mask"].to(device)
    audio_mask = prepared["audio_target_mask"].to(device)
    id_mask = prepared["id_target_mask"].to(device)
    boundary_mask = prepared["boundary_target_mask"].to(device)
    degraded = torch.arange(documents, device=device).remainder(2).bool()

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(input_ids, attention_mask)
        base_loss, _ = degraded_causal_base_losses(
            logits,
            input_ids,
            audio_mask,
            id_mask,
            boundary_mask,
            degraded,
            id_digit_weight=32.0,
        )
        distillation = identifier_logit_distillation_loss(
            logits,
            id_mask,
            degraded,
            prepared["track_id"],
            list(tokenizer.vocabulary.digit_token_ids),
            temperature=2.0,
        )
        loss = base_loss + 0.1 * distillation
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()


def _probe_candidate(
    *,
    tracks: int,
    config: dict,
    tokenizer: MuQRVQTokenizer,
    device: torch.device,
) -> dict:
    documents = tracks * DOCUMENTS_PER_TRACK
    base_model = None
    model = None
    optimizer = None
    waveforms = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        base_model = AudioCausalLM(_model_config(config), tokenizer.vocabulary).to(
            device
        )
        model = DistributedDataParallel(
            base_model,
            device_ids=[device.index],
            output_device=device.index,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=3.0e-4,
            betas=(0.9, 0.95),
            weight_decay=0.0,
        )
        waveforms = torch.randn(
            documents,
            48_000,
            device=device,
            dtype=torch.float32,
        ).mul_(0.05)
        for _ in range(WARMUP_UPDATES):
            _optimizer_update(
                model=model,
                optimizer=optimizer,
                tokenizer=tokenizer,
                waveforms=waveforms,
                device=device,
            )
        torch.cuda.synchronize(device)
        timed_started = time.perf_counter()
        for _ in range(TIMED_UPDATES):
            _optimizer_update(
                model=model,
                optimizer=optimizer,
                tokenizer=tokenizer,
                waveforms=waveforms,
                device=device,
            )
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - timed_started
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        peak_headroom = max(0.0, 100.0 * (total_bytes - peak_reserved) / total_bytes)
        mean_step_seconds = elapsed / TIMED_UPDATES
        return {
            "tracks_per_gpu": tracks,
            "documents_per_gpu": documents,
            "success": True,
            "peak_allocated_gib": round(
                torch.cuda.max_memory_allocated(device) / 2**30, 3
            ),
            "peak_reserved_gib": round(
                peak_reserved / 2**30, 3
            ),
            "peak_headroom_percent": round(peak_headroom, 2),
            "free_after_step_gib": round(free_bytes / 2**30, 3),
            "free_after_step_percent": round(100.0 * free_bytes / total_bytes, 2),
            "total_memory_gib": round(total_bytes / 2**30, 3),
            "warmup_updates": WARMUP_UPDATES,
            "timed_updates": TIMED_UPDATES,
            "mean_step_seconds": round(mean_step_seconds, 4),
            "documents_per_second": round(documents / mean_step_seconds, 3),
        }
    except (RuntimeError, torch.OutOfMemoryError) as error:
        if not _is_oom(error):
            raise
        return {
            "tracks_per_gpu": tracks,
            "documents_per_gpu": documents,
            "success": False,
            "error": "CUDA out of memory",
        }
    finally:
        del waveforms, optimizer, model, base_model
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe tc18 medium per-GPU training memory without creating a run."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--tracks",
        type=int,
        nargs="+",
        default=list(DEFAULT_CANDIDATES),
        help="ascending per-GPU track counts to test",
    )
    args = parser.parse_args()
    candidates = validate_candidates(args.tracks)
    if not torch.cuda.is_available():
        raise RuntimeError("The medium memory probe requires a CUDA GPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Run this diagnostic in a one-GPU session")

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29571")
    torch.distributed.init_process_group("nccl", rank=0, world_size=1)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    config = load_config(args.config)
    tokenizer_cfg = config["tokenizer"]
    tokenizer = MuQRVQTokenizer(
        tokenizer_cfg["model_name"],
        revision=tokenizer_cfg.get("revision", "main"),
        selected_codebooks=8,
        sample_rate=24_000,
        device=device,
        lightweight=True,
    )
    results = []
    try:
        for tracks in candidates:
            result = _probe_candidate(
                tracks=tracks,
                config=config,
                tokenizer=tokenizer,
                device=device,
            )
            results.append(result)
            print(json.dumps(result), flush=True)
            if not result["success"]:
                break
    finally:
        torch.distributed.destroy_process_group()

    recommendation = recommend_candidate(
        results, minimum_headroom_percent=MINIMUM_HEADROOM_PERCENT
    )
    print(json.dumps({"recommendation": recommendation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
