"""Small paired clean / training-IR / held-out-IR identification diagnostic.

Run from the repository root with python -m ablations.rir_generalization.
Training-pool IRs are not proof that a specific track/IR pair was seen.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random

import numpy as np
import torch
from tqdm import tqdm

from para_audio_id.audio import load_audio
from para_audio_id.catalogue import load_catalogue
from para_audio_id.audio_lm.checkpoint import load_audio_lm
from para_audio_id.audio_lm.evaluation import (
    _checkpoint_file_fingerprint,
    _checkpoint_tokenizer,
    _load_query_context,
    _valid_waveform,
)
from para_audio_id.audio_lm.generation import (
    batched_beam_generate,
    prompts_from_audio_tokens,
)
from para_audio_id.audio_lm.rir import RoomImpulseResponseAssets, convolve_full_wet


def severity_match(files, target, scores, rng):
    """Randomize ties, then choose the closest available decay duration."""
    candidates = list(files)
    rng.shuffle(candidates)
    return min(candidates, key=lambda path: abs(scores[path] - target))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("rir-generalization.json"))
    parser.add_argument("--tracks", type=int, default=100)
    parser.add_argument("--irs-per-track", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if min(args.tracks, args.irs_per_track, args.batch_size) < 1:
        parser.error("Counts must be positive")
    if args.output.exists():
        parser.error("Output exists; choose a new --output filename")

    # CPU loading avoids putting checkpoint optimizer tensors on the GPU.
    model, vocabulary, cfg, checkpoint = load_audio_lm(args.checkpoint, "cpu")
    model.to(args.device).eval()
    print("Validating room-IR pools...", flush=True)
    rate = int(checkpoint["tokenizer_spec"]["sample_rate"])
    duration = float(cfg["data"]["segment_duration"])
    samples = round(rate * duration)
    rir_cfg = cfg["data"]["room_ir"]
    past = round(rate * float(rir_cfg.get("past_context_duration", 2.0)))
    assets = RoomImpulseResponseAssets(
        rir_cfg["training_root"], rir_cfg["validation_root"], sample_rate=rate
    )
    if checkpoint.get("room_ir_manifest") != assets.manifest():
        raise ValueError("IR manifests differ from the checkpoint")
    all_files = assets.training_files + assets.validation_files
    scores = {path: assets._severity_score(path) for path in all_files}
    train_files = [p for p in assets.training_files if np.isfinite(scores[p])]
    test_files = [p for p in assets.validation_files if np.isfinite(scores[p])]
    if not train_files or not test_files:
        raise ValueError("Both IR pools need valid severity measurements")

    records = {r.track_id: r for r in load_catalogue(cfg["data"]["catalogue"])}
    ids = sorted(checkpoint["training_track_ids"])
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    audio_root = Path(cfg["data"]["audio_root"])
    tokenizer = _checkpoint_tokenizer(checkpoint, args.device)
    rows, exclusions, pending = [], [], []

    def flush():
        if not pending:
            return
        waveforms = torch.from_numpy(np.stack([w for _, w in pending])).to(args.device)
        with torch.inference_mode():
            tokens = tokenizer.tokenize(waveforms)
            expected = round(duration * tokenizer.spec.frame_rate) * vocabulary.num_codebooks
            if tokens.shape[1] != expected:
                raise ValueError(f"Unexpected token shape: {tokens.shape}")
            prompts = prompts_from_audio_tokens(tokens, vocabulary)
            autocast = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if args.device.startswith("cuda") else nullcontext()
            )
            with autocast:
                rankings = batched_beam_generate(
                    model, prompts, vocabulary, width=10, score_eos=False
                )
        for (recipe, _), ranking in zip(pending, rankings, strict=True):
            codes = [candidate.code for candidate in ranking]
            rows.append({**recipe, "top1": codes[0] == recipe["code"], "ranking": codes})
        pending.clear()

    selected = 0
    for track_id in tqdm(ids, desc="paired RIR diagnostic"):
        if selected == args.tracks:
            break
        record = records[track_id]
        start = rng.randrange(max(1, int(record.duration * rate) - samples + 1))
        recipe = {
            "track_id": track_id, "code": record.code, "source_path": record.path,
            "start_sample": start, "duration": duration,
        }
        try:
            context = _load_query_context(
                audio_root / record.path, sample_rate=rate, start_sample=start,
                query_samples=samples, past_samples=past,
            )
            paired = [({**recipe, "condition": "clean"}, context[past:])]
            for pair_index in range(args.irs_per_track):
                training_ir = rng.choice(train_files)
                heldout_ir = severity_match(test_files, scores[training_ir], scores, rng)
                for condition, path in (("seen", training_ir), ("unseen", heldout_ir)):
                    ir = load_audio(path, sample_rate=rate, duration=None)
                    waveform = convolve_full_wet(
                        context, ir, past_context_samples=past, output_samples=samples
                    )
                    _valid_waveform(waveform, samples)
                    paired.append(({
                        **recipe, "condition": condition, "pair": pair_index,
                        "ir_path": str(path), "decay_seconds": scores[path],
                        "decay_mismatch_seconds": abs(scores[training_ir] - scores[heldout_ir]),
                    }, waveform))
        except (ValueError, RuntimeError, OSError) as error:
            exclusions.append({**recipe, "error": str(error)})
            continue
        selected += 1
        for item in paired:
            pending.append(item)
            if len(pending) == args.batch_size:
                flush()
    flush()
    if selected != args.tracks:
        raise RuntimeError(f"Only {selected}/{args.tracks} usable tracks")
    metrics = {}
    for condition in ("clean", "seen", "unseen"):
        subset = [row for row in rows if row["condition"] == condition]
        metrics[condition] = {
            "queries": len(subset),
            "top1": sum(row["top1"] for row in subset) / len(subset),
        }
    payload = {
        "protocol": "paired_training_vs_heldout_rir_v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _checkpoint_file_fingerprint(args.checkpoint),
        "checkpoint_step": checkpoint.get("global_step"),
        "tokenizer_fingerprint": checkpoint["tokenizer_fingerprint"],
        "ir_manifest": assets.manifest(), "seed": args.seed,
        "tracks": selected, "irs_per_track": args.irs_per_track,
        "beam_width": 10, "score_eos": False,
        "seen_definition": "training pool, not verified historical track/IR pair",
        "severity_measure": "post_peak_99_percent_energy_decay_seconds",
        "metrics": metrics, "exclusions": exclusions, "queries": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print("\nCondition       Queries    Top-1")
    for condition, metric in metrics.items():
        print(f"{condition:15s} {metric['queries']:6d}    {100 * metric['top1']:.2f}%")
    print(f"Excluded candidates: {len(exclusions)}; saved {args.output}")


if __name__ == "__main__":
    main()
