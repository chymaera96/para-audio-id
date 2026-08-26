from __future__ import annotations

import argparse
import gc
import json
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


DEFAULT_CANDIDATES = (1, 2, 4, 5, 8, 10)
GLOBAL_TRACKS_PER_STEP = 80
DOCUMENTS_PER_TRACK = 2


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
    prepared = None
    logits = None
    loss = None
    started = time.perf_counter()
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
        audio_tokens = tokenizer.tokenize(waveforms)
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
        track_ids = prepared["track_id"]

        # Two updates are required: AdamW allocates its state on the first step.
        for _ in range(2):
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
                    track_ids,
                    list(tokenizer.vocabulary.digit_token_ids),
                    temperature=2.0,
                )
                loss = base_loss + 0.1 * distillation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        torch.cuda.synchronize(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        return {
            "tracks_per_gpu": tracks,
            "documents_per_gpu": documents,
            "success": True,
            "peak_allocated_gib": round(
                torch.cuda.max_memory_allocated(device) / 2**30, 3
            ),
            "peak_reserved_gib": round(
                torch.cuda.max_memory_reserved(device) / 2**30, 3
            ),
            "free_after_step_gib": round(free_bytes / 2**30, 3),
            "total_memory_gib": round(total_bytes / 2**30, 3),
            "seconds": round(time.perf_counter() - started, 3),
        }
    except (RuntimeError, torch.OutOfMemoryError) as error:
        if not _is_oom(error):
            raise
        return {
            "tracks_per_gpu": tracks,
            "documents_per_gpu": documents,
            "success": False,
            "error": "CUDA out of memory",
            "seconds": round(time.perf_counter() - started, 3),
        }
    finally:
        del loss, logits, prepared, waveforms, optimizer, model, base_model
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
    candidates = list(dict.fromkeys(args.tracks))
    if candidates != sorted(candidates) or any(
        value <= 0 or (GLOBAL_TRACKS_PER_STEP // 2) % value for value in candidates
    ):
        raise ValueError(
            "--tracks must be unique ascending positive divisors of 40"
        )
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

    successful = [row for row in results if row["success"]]
    if not successful:
        raise RuntimeError("No tested medium microbatch fit on this GPU")
    best = successful[-1]
    tracks = int(best["tracks_per_gpu"])
    recommendation = {
        "largest_successful_tracks_per_gpu": tracks,
        "documents_per_gpu": tracks * DOCUMENTS_PER_TRACK,
        "two_gpu_accumulate_grad_batches": (
            GLOBAL_TRACKS_PER_STEP // (2 * tracks)
        ),
        "global_tracks_per_optimizer_step": GLOBAL_TRACKS_PER_STEP,
        "note": "Choose the next smaller tested layout if peak memory has little headroom.",
    }
    print(json.dumps({"recommendation": recommendation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
