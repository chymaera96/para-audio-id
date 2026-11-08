"""Shared fresh-crop probes outside fixed replay observations."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from ..catalogue import CatalogueRecord
from .checkpoint import load_audio_lm
from .continual import (CONDITIONS, asset_loaders, digest, evaluate_tokens,
                        file_digest, write_json)
from .continual_replay import ReplayCache
from .evaluation import _checkpoint_tokenizer
from .noise import mix_background_noise, stable_uint64
from .random_crops import RandomEvaluationCollator
from .rir import convolve_full_wet


def load_fresh(directory, pilot):
    directory = Path(directory)
    meta = json.loads((directory / "manifest.json").read_text())
    if meta["protocol"] != "continual_fresh_queries_v1" or meta["pilot_sha256"] != digest(pilot):
        raise ValueError("Fresh evaluation pilot mismatch")
    if file_digest(directory / "queries.pt") != meta["queries_sha256"]:
        raise ValueError("Fresh evaluation token file changed")
    if file_digest(directory / "baseline.json") != meta["baseline_sha256"]:
        raise ValueError("Fresh evaluation baseline changed")
    data = torch.load(directory / "queries.pt", map_location="cpu", weights_only=False)
    if data["tokenizer_fingerprint"] != pilot["tokenizer_fingerprint"]:
        raise ValueError("Fresh evaluation tokenizer mismatch")
    return data


def prepare_fresh(pilot_directory, output, *, count=1000):
    from tqdm import tqdm

    started = time.perf_counter()
    pilot_directory, output = Path(pilot_directory), Path(output)
    pilot = json.loads((pilot_directory / "manifest.json").read_text())
    if (output / "manifest.json").exists():
        load_fresh(output, pilot)
        meta = json.loads((output / "manifest.json").read_text())
        if meta["tracks_per_group"] != count:
            raise ValueError("Existing fresh query count differs")
        print("Using verified fresh query pack", output, flush=True)
        return
    if file_digest(pilot["source_checkpoint"]) != pilot["source_sha256"]:
        raise ValueError("Source checkpoint changed")
    output.mkdir(parents=True, exist_ok=True)
    replay = ReplayCache(pilot_directory / "replay-cache-v1", pilot)
    forbidden = defaultdict(set)
    for track, starts in pilot["reserved_starts"].items():
        forbidden[track].update(starts)
    for row in replay.rows:
        forbidden[row["track_id"]].add(row["start_sample"])
    model, vocabulary, cfg, source = load_audio_lm(pilot["source_checkpoint"], "cpu")
    model.cuda().eval()
    tokenizer = _checkpoint_tokenizer(source, "cuda")
    sr = tokenizer.sample_rate
    noise, room = asset_loaders(cfg, sr)
    if noise.manifest() != pilot["noise_manifest"] or room.manifest() != pilot["rir_manifest"]:
        raise ValueError("Degradation assets changed")
    seed = pilot["seed"] + 4200
    evaluator = RandomEvaluationCollator(audio_root=cfg["data"]["audio_root"],
        noise_assets=noise, rir_assets=room, sample_rate=sr, past_context_duration=2.0, seed=seed)
    batches, rows, rejected, reserved = [], [], [], {}
    rng = np.random.default_rng(seed)
    for group in ("old", "new"):
        pool = [CatalogueRecord(**r) for r in pilot[group]]
        accepted = 0
        for index in tqdm(rng.permutation(len(pool)), desc=f"Fresh {group} tracks", total=count):
            record = pool[int(index)]
            maximum = max(0, round(record.duration * sr) - 2 * sr)
            for attempt in range(8):
                start = int(stable_uint64(seed, record.track_id, attempt, "fresh-start") % (maximum + 1))
                if start in forbidden[record.track_id]:
                    continue
                recipe = dict(track_id=record.track_id, code=record.code, source_path=record.path,
                    source_duration=record.duration, start_sample=start, start=start / sr,
                    crop_duration=2.0, view_type="continual_fresh_heldout")
                try:
                    batch = evaluator([recipe])
                    if batch["skipped"]:
                        raise ValueError(str(batch["skipped"]))
                    noisy, valid = mix_background_noise(batch["clean_waveforms"],
                        batch["noise_waveforms"], torch.tensor([0.0]))
                    context, context_noise, ir = batch["noise_rir_inputs"][0]
                    mixed, context_valid = mix_background_noise(torch.from_numpy(context)[None],
                        torch.from_numpy(context_noise)[None], torch.tensor([0.0]))
                    if not bool(valid.all() and context_valid.all()):
                        raise ValueError("Invalid fresh query SNR")
                    combined = convolve_full_wet(mixed[0].numpy(), ir,
                        past_context_samples=2 * sr, output_samples=2 * sr)
                    waves = torch.cat([batch["clean_waveforms"], noisy, torch.from_numpy(combined)[None]])
                    if not torch.isfinite(waves).all() or (waves.square().mean(1) < 1e-12).any():
                        raise ValueError("Invalid fresh waveform")
                    with torch.inference_mode():
                        tokens = tokenizer.tokenize(waves.cuda()).cpu().to(torch.int16)
                    batches.append(tokens)
                    rows.extend(dict(group=group, condition=condition, **recipe) for condition in CONDITIONS)
                    reserved[record.track_id] = [start]
                    accepted += 1
                    break
                except Exception as exc:
                    rejected.append(dict(track_id=record.track_id, attempt=attempt, error=str(exc)))
            if accepted == count:
                break
        if accepted != count:
            raise RuntimeError(f"Only {accepted}/{count} usable fresh {group} queries")
    cache = dict(tokens=torch.cat(batches), rows=rows, tokenizer_fingerprint=pilot["tokenizer_fingerprint"])
    torch.save(cache, output / "queries.pt")
    metrics, details = evaluate_tokens(model, vocabulary, cache, "cuda")
    write_json(output / "baseline.json", dict(metrics=metrics, rows=details, source_sha256=pilot["source_sha256"]))
    meta = dict(protocol="continual_fresh_queries_v1", pilot_sha256=digest(pilot),
        replay_sha256=replay.sha256, tracks_per_group=count, seed=seed,
        reserved_starts=reserved, rejected=rejected,
        queries_sha256=file_digest(output / "queries.pt"),
        baseline_sha256=file_digest(output / "baseline.json"),
        preparation_seconds=time.perf_counter() - started)
    write_json(output / "manifest.json", meta)
    print("FRESH PACK READY", metrics, flush=True)
