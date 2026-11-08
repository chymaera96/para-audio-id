"""Shared data and evaluation for the tc21 25K -> 26K indexing pilot.

Independent of ordinary training/resume: the source checkpoint is immutable.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Sampler

from ..audio import load_audio
from ..catalogue import load_catalogue
from .checkpoint import load_audio_lm
from .evaluation import _checkpoint_tokenizer
from .generation import batched_beam_generate, prompts_from_audio_tokens
from .noise import BackgroundNoiseAssets, mix_background_noise, stable_uint64
from .random_crops import RandomCropCollator, RandomEvaluationCollator
from .rir import RoomImpulseResponseAssets, convolve_full_wet

PROTOCOL = "fsq_continual_additions_v1"
CONDITIONS = ("clean", "noise_0db", "noise_rir_0db")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def asset_loaders(cfg, sample_rate):
    data = cfg["data"]
    noise = BackgroundNoiseAssets(data["background_noise"]["training_root"],
                                  data["background_noise"]["validation_root"],
                                  sample_rate=sample_rate, samples=2 * sample_rate)
    room = RoomImpulseResponseAssets(data["room_ir"]["training_root"],
                                     data["room_ir"]["validation_root"],
                                     sample_rate=sample_rate)
    return noise, room


def full_schedule(source):
    schedule = dict(source["schedule_config"])
    schedule.pop("ramp_steps", None)
    schedule.update(curriculum="tc12_noise_rir_curriculum_v1", clean_until_step=0,
                    degradation_ramp_until_step=0, combined_ramp_until_step=0)
    return schedule


def valid_record(record, root, sr, seed):
    maximum = max(0, round(record.duration * sr) - 2 * sr)
    start = stable_uint64(seed, record.track_id, "new-track-audit") % (maximum + 1)
    try:
        wave = load_audio(Path(root) / record.path, sample_rate=sr,
                          start=start / sr, duration=2.0, pad=True)
        if len(wave) != 2 * sr or not np.isfinite(wave).all() or np.mean(wave ** 2) < 1e-12:
            raise ValueError("silent or invalid crop")
        return None
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def choose_groups(records, old_ids, *, new_count, buffer_count, seed, validator):
    by_id = {r.track_id: r for r in records}
    if len(by_id) != len(records) or len(set(old_ids)) != len(old_ids):
        raise ValueError("Duplicate catalogue or source identities")
    old = [by_id[track] for track in old_ids]
    old_set = set(old_ids)
    candidates = sorted((r for r in records if r.track_id not in old_set),
                        key=lambda r: r.track_id)
    rng = np.random.default_rng(seed)
    candidates = [candidates[i] for i in rng.permutation(len(candidates))]
    new, rejected = [], []
    # Validate only enough candidates; preserve deterministic order across workers.
    with ThreadPoolExecutor(max_workers=6) as executor:
        for start in range(0, len(candidates), 32):
            chunk = candidates[start:start + 32]
            for record, error in zip(chunk, executor.map(validator, chunk), strict=True):
                if error is None:
                    new.append(record)
                else:
                    rejected.append({"track_id": record.track_id, "error": error})
                if len(new) == new_count:
                    break
            if len(new) == new_count:
                break
    if len(new) != new_count or buffer_count >= len(old):
        raise ValueError("Insufficient valid newcomers or old replay identities")
    buffer = rng.choice(old_ids, size=buffer_count, replace=False).tolist()
    return old, new, buffer, rejected


@torch.inference_mode()
def evaluate_tokens(model, vocabulary, cache, device, *, batch_size=16):
    model.eval()
    predictions = []
    for offset in range(0, len(cache["rows"]), batch_size):
        tokens = cache["tokens"][offset:offset + batch_size].to(device).long()
        beams = batched_beam_generate(model, prompts_from_audio_tokens(tokens, vocabulary),
                                      vocabulary, width=10, score_eos=True)
        predictions.extend(b[0].code for b in beams)
    metrics, details = {}, []
    for row, prediction in zip(cache["rows"], predictions, strict=True):
        details.append({**row, "prediction": prediction, "correct": prediction == row["code"]})
    for group in ("old", "new"):
        for condition in CONDITIONS:
            rows = [r for r in details if r["group"] == group and r["condition"] == condition]
            if not rows:
                raise ValueError("Incomplete old/new validation cache")
            metrics[f"val/{group}/{condition}"] = sum(r["correct"] for r in rows) / len(rows)
    return metrics, details


def prepare(source_path, output, *, new_count=1000, buffer_count=1000,
            monitor_count=100, seed=7331, device="cuda"):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Use the existing prepared pilot or a new directory: {output}")
    output.mkdir(parents=True)
    model, vocabulary, cfg, source = load_audio_lm(source_path, "cpu")
    if source["tokenizer_spec"]["architecture"] != "cnn_fsq_tokenizer_v1":
        raise ValueError("This pilot requires a frozen FSQ source")
    sr = int(source["tokenizer_spec"]["sample_rate"])
    records = load_catalogue(cfg["data"]["catalogue"])
    old, new, buffer, rejected = choose_groups(
        records, source["training_track_ids"], new_count=new_count,
        buffer_count=buffer_count, seed=seed,
        validator=lambda r: valid_record(r, cfg["data"]["audio_root"], sr, seed))
    original = {r.track_id: r.code for r in old}
    expected = hashlib.sha256("\n".join(f"{r.track_id}:{r.code}" for r in old).encode()).hexdigest()
    if expected != source["code_mapping_fingerprint"]:
        raise ValueError("Source catalogue identifier assignments have changed")
    if set(original.values()) & {r.code for r in new}:
        raise ValueError("New identifiers collide with old identifiers")
    noise, room = asset_loaders(cfg, sr)
    tokenizer = _checkpoint_tokenizer(source, device)
    evaluator = RandomEvaluationCollator(audio_root=cfg["data"]["audio_root"],
        noise_assets=noise, rir_assets=room, sample_rate=sr,
        past_context_duration=2.0, seed=seed)
    rows, token_batches, reserved, val_audit = [], [], {}, []
    rng = np.random.default_rng(seed + 1)
    buffer_set = set(buffer)
    for group, pool in [("old", [r for r in old if r.track_id not in buffer_set]), ("new", new)]:
        accepted = 0
        for i in rng.permutation(len(pool)):
            record = pool[int(i)]
            start = stable_uint64(seed, record.track_id, "validation-start") % (
                max(0, round(record.duration * sr) - 2 * sr) + 1)
            recipe = dict(track_id=record.track_id, code=record.code, source_path=record.path,
                          source_duration=record.duration, start_sample=int(start), start=start / sr,
                          crop_duration=2.0, view_type="continual_heldout")
            batch = evaluator([recipe])
            if batch["skipped"]:
                val_audit.extend(batch["skipped"])
                continue
            try:
                noisy, valid = mix_background_noise(batch["clean_waveforms"],
                    batch["noise_waveforms"], torch.tensor([0.0]))
                context, context_noise, ir = batch["noise_rir_inputs"][0]
                mixed, context_valid = mix_background_noise(torch.from_numpy(context)[None],
                    torch.from_numpy(context_noise)[None], torch.tensor([0.0]))
                if not bool(valid.all() and context_valid.all()):
                    raise ValueError("Invalid validation SNR")
                combined = convolve_full_wet(mixed[0].numpy(), ir,
                    past_context_samples=2 * sr, output_samples=2 * sr)
                waveforms = torch.cat([batch["clean_waveforms"], noisy,
                                       torch.from_numpy(combined)[None]])
                if not torch.isfinite(waveforms).all() or (waveforms.square().mean(1) < 1e-12).any():
                    raise ValueError("Invalid validation waveform")
                with torch.inference_mode():
                    tokens = tokenizer.tokenize(waveforms.to(device)).cpu().to(torch.int16)
            except Exception as exc:
                val_audit.append({"track_id": record.track_id, "error": str(exc)})
                continue
            token_batches.append(tokens)
            rows.extend(dict(group=group, condition=condition, **recipe) for condition in CONDITIONS)
            reserved[record.track_id] = [int(start)]
            accepted += 1
            if accepted == monitor_count:
                break
        if accepted != monitor_count:
            raise RuntimeError(f"Could only prepare {accepted} {group} validation identities")
    cache = {"tokens": torch.cat(token_batches), "rows": rows,
             "tokenizer_fingerprint": source["tokenizer_fingerprint"]}
    torch.save(cache, output / "validation.pt")
    manifest = dict(protocol=PROTOCOL, source_checkpoint=str(Path(source_path).resolve()),
        source_sha256=file_digest(source_path), source_global_step=int(source["global_step"]),
        tokenizer_fingerprint=source["tokenizer_fingerprint"], seed=seed,
        old=[asdict(r) for r in old], new=[asdict(r) for r in new], replay_ids=buffer,
        reserved_starts=reserved, rejected_new=rejected, validation_audit=val_audit,
        validation_sha256=file_digest(output / "validation.pt"),
        noise_manifest=noise.manifest(), rir_manifest=room.manifest())
    model.to(device)
    metrics, details = evaluate_tokens(model, vocabulary, cache, device)
    write_json(output / "baseline.json", {"metrics": metrics, "rows": details,
                                         "source_sha256": manifest["source_sha256"]})
    manifest["baseline_sha256"] = file_digest(output / "baseline.json")
    write_json(output / "manifest.json", manifest)
    print(f"Prepared {len(old)} old + {len(new)} new tracks; {len(rows)} validation queries.", flush=True)
    print("Source baseline:", metrics, flush=True)
    return manifest


class AdditionBatchSampler(Sampler):
    """Stateless update-indexed sampling, including exact resume and B/C pairing."""
    def __init__(self, new_count, old_count, *, arm, steps, start=0,
                 microbatch=40, accumulation=2, seed=7331):
        self.new_count, self.old_count = new_count, old_count
        self.arm, self.steps, self.start = arm, steps, start
        self.microbatch, self.accumulation, self.seed = microbatch, accumulation, seed
        if arm not in "ABCDE" or len(arm) != 1 or microbatch % 2:
            raise ValueError("Use arm A/B/C/D/E and an even microbatch")

    def __len__(self):
        return (self.steps - self.start) * self.accumulation

    def __iter__(self):
        for step in range(self.start, self.steps):
            for micro in range(self.accumulation):
                count = self.microbatch if self.arm == "A" else self.microbatch // 2
                rng = np.random.default_rng(stable_uint64(self.seed, step, micro, "new"))
                selected = rng.choice(self.new_count, count, replace=False).tolist()
                if self.arm != "A":
                    rng = np.random.default_rng(stable_uint64(self.seed, step, micro, "old"))
                    selected += (self.new_count + rng.choice(
                        self.old_count, self.microbatch - count, replace=False)).tolist()
                yield [(int(i), step, step * self.accumulation + micro, slot)
                       for slot, i in enumerate(selected)]


class AdditionCollator:
    """Use separate replacement pools so corrupt audio cannot change old/new ratios."""
    def __init__(self, new, old, **kwargs):
        self.pools = [new, old]
        self.collators = [RandomCropCollator(records=pool, **kwargs) if pool else None
                          for pool in self.pools]
        self.maps = [{r.track_id: i for i, r in enumerate(pool)} for pool in self.pools]

    def __call__(self, examples):
        parts = []
        for pool, collator in zip(self.maps, self.collators, strict=True):
            selected = [{**e, "record_index": pool[e["record"].track_id]}
                        for e in examples if e["record"].track_id in pool]
            if selected:
                parts.append(collator(selected))
        return {"waveforms": torch.cat([p["waveforms"] for p in parts]),
                "metadata": [r for p in parts for r in p["metadata"]],
                "replacements": [r for p in parts for r in p["replacements"]],
                "failures": [r for p in parts for r in p["failed_crop_attempts"]],
                "categories": dict(sum((Counter(p["categories"]) for p in parts), Counter()))}
