"""Reproducible FSQ source split and paired crops using PSI's degradations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..audio import load_audio
from .noise import BackgroundNoiseAssets, mix_background_noise, stable_uint64
from .rir import RoomImpulseResponseAssets, convolve_full_wet


CONDITIONS = ("clean", "noise", "rir", "noise_rir")


def source_split(cfg):
    path = Path(cfg["data"]["split_path"])
    catalogue = Path(cfg["data"]["catalogue"])
    digest = hashlib.sha256(catalogue.read_bytes()).hexdigest()
    data = cfg["data"]
    identity = dict(seed=cfg["seed"], catalogue_sha256=digest,
                    training_count=data["training_tracks"], monitor_count=data["monitor_tracks"])
    if path.exists():
        split = json.loads(path.read_text())
        if any(split.get(k) != v for k, v in identity.items()):
            raise ValueError(f"Existing split differs from configuration: {path}")
        if "monitor_crop_keys" not in split:
            prepare_monitor(cfg, split)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(split, indent=2))
            temporary.replace(path)
        return split
    records = [json.loads(line) for line in catalogue.read_text().splitlines() if line.strip()]
    records = sorted(records, key=lambda r: r["track_id"])
    if len({r["track_id"] for r in records}) != len(records):
        raise ValueError("Duplicate catalogue identities")
    n, m = data["training_tracks"], data["monitor_tracks"]
    if n + m > len(records):
        raise ValueError("Not enough source tracks for disjoint train/monitor split")
    rng = np.random.default_rng(cfg["seed"])
    selected = rng.permutation(len(records))[:n + m]
    split = {**identity, "training": [records[i] for i in selected[:n]],
             "monitor": [records[i] for i in selected[n:]]}
    prepare_monitor(cfg, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create prevents an accidental overwrite by another preparation.
    with path.open("x") as f:
        json.dump(split, f, indent=2)
    return split


def read_source(cfg, record, key):
    d = cfg["data"]
    rng = np.random.default_rng(key)
    sr = d["sample_rate"]
    samples = round(d["crop_duration"] * sr)
    past = round(d["past_context_duration"] * sr)
    maximum = max(0, round(record["duration"] * sr) - samples)
    start = int(rng.integers(maximum + 1))
    available = min(start, past)
    context = load_audio(Path(d["audio_root"]) / record["path"], sample_rate=sr,
                         start=(start - available) / sr,
                         duration=(available + samples) / sr, pad=True)
    context = np.pad(context, (past - available, 0))
    if not np.isfinite(context).all() or np.square(context[-samples:]).mean() <= 1e-10:
        raise ValueError("Invalid or silent source crop")
    return context, start, rng


def prepare_monitor(cfg, split):
    """Choose usable fixed crops once; persist any source replacement decisions."""
    records = [json.loads(line) for line in Path(cfg["data"]["catalogue"]).read_text().splitlines()
               if line.strip()]
    reserved = {r["track_id"] for r in split["training"] + split["monitor"]}
    candidates = sorted((r for r in records if r["track_id"] not in reserved),
                        key=lambda r: stable_uint64(cfg["seed"], r["track_id"], "monitor-replacement"))
    replacements = iter(candidates)
    keys, audit = [], []
    for index, original in enumerate(split["monitor"]):
        record = original
        while True:
            for attempt in range(8):
                key = stable_uint64(cfg["seed"], index, "fixed-monitor-crop") + attempt
                try:
                    read_source(cfg, record, key)
                    break
                except (ValueError, RuntimeError, OSError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
            else:
                audit.append({"track_id": record["track_id"], "path": record["path"],
                              "error": error, "action": "replace_monitor_track"})
                record = next(replacements)
                continue
            if attempt:
                audit.append({"track_id": record["track_id"], "action": "retry_monitor_crop",
                              "attempt": attempt})
            split["monitor"][index] = record
            keys.append(key)
            break
    split["monitor_crop_keys"] = keys
    split["monitor_preparation_audit"] = audit


def assets(cfg):
    d = cfg["data"]
    noise = BackgroundNoiseAssets(**d["background_noise"], sample_rate=d["sample_rate"],
                                  samples=round(d["crop_duration"] * d["sample_rate"]))
    rir = RoomImpulseResponseAssets(**d["room_ir"], sample_rate=d["sample_rate"])
    return noise, rir


class PairedCrops(Dataset):
    def __init__(self, cfg, records, noise, rir, *, monitor=False, start_index=0,
                 monitor_keys=None):
        self.cfg, self.records, self.noise, self.rir = cfg, records, noise, rir
        self.monitor, self.start_index = monitor, start_index
        self.monitor_keys = monitor_keys

    def __len__(self):
        if self.monitor:
            return len(self.records) * 4
        return self.cfg["train"]["max_steps"] * self.cfg["train"]["batch_size"] - self.start_index

    def __getitem__(self, index):
        d = self.cfg["data"]
        index += self.start_index
        key = stable_uint64(self.cfg["seed"], index, "monitor" if self.monitor else "training")
        rng = np.random.default_rng(key)
        if self.monitor:
            record_index, condition = divmod(index, 4)
            # All four conditions use the same clean crop and augmentation recipe.
            key = (self.monitor_keys[record_index] if self.monitor_keys is not None else
                   stable_uint64(self.cfg["seed"], record_index, "fixed-monitor-crop"))
        else:
            record_index = int(rng.integers(len(self.records)))
            condition = int(rng.choice(4, p=d["probabilities"]))
        for attempt in range(8):
            record = self.records[(record_index + attempt) % len(self.records)]
            try:
                return self.pair(record, key + attempt, condition, attempt)
            except (ValueError, RuntimeError, OSError) as exc:
                if self.monitor:
                    raise RuntimeError(f"Fixed monitoring track failed: {record['path']}") from exc
                if attempt == 7:
                    raise RuntimeError(f"No usable crop after 8 attempts; last: {record['path']}") from exc
        raise AssertionError("unreachable")

    def pair(self, record, key, condition, replacements):
        d = self.cfg["data"]
        sr = d["sample_rate"]
        samples = round(d["crop_duration"] * sr)
        past = round(d["past_context_duration"] * sr)
        context, start, rng = read_source(self.cfg, record, key)
        clean = context[-samples:].copy()
        degraded_context = context.copy()
        loader = self.noise.load_validation if self.monitor else self.noise.load_training
        if condition in (1, 3):
            bin_index = int(rng.choice(4, p=d["snr_bin_probabilities"]))
            low, high = d["snr_bins"][bin_index]
            snr = float(rng.uniform(low, high))
            if bin_index == 0 and rng.random() < d["exact_zero_fraction_in_first_bin"]:
                snr = 0.0
            noise = loader(key, samples=len(context))
            if condition == 1:
                mixed, valid = mix_background_noise(torch.from_numpy(clean[None]),
                    torch.from_numpy(noise[-samples:][None]), torch.tensor([snr]))
                degraded = mixed[0].numpy()
            else:
                mixed, valid = mix_background_noise(torch.from_numpy(context[None]),
                    torch.from_numpy(noise[None]), torch.tensor([snr]))
                degraded_context = mixed[0].numpy()
            if not bool(valid.all()):
                raise ValueError("Silent noise or source")
        if condition in (2, 3):
            loader = self.rir.load_validation if self.monitor else self.rir.load_training
            ir, _ = loader(key)
            degraded = convolve_full_wet(degraded_context, ir, past_context_samples=past,
                                         output_samples=samples)
        elif condition == 0:
            degraded = clean.copy()
        return {"clean": torch.from_numpy(clean), "degraded": torch.from_numpy(degraded),
                "condition": condition, "replacements": replacements,
                "track_id": record["track_id"], "start_sample": start}
