from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from .audio import BadFileRegistry, load_audio, quantile_normalize
from .augment import WaveformAugmenter
from .catalogue import load_catalogue
from .codes import code_to_tokens


class CatalogueCropDataset(Dataset):
    def __init__(self, cfg: dict, *, training: bool):
        self.cfg = cfg
        self.training = training
        data = cfg["data"]
        self.root = Path(data["audio_root"])
        self.records = load_catalogue(data["catalogue"])
        self.sample_rate = int(cfg["model"]["sample_rate"])
        self.duration = float(data["query_duration"])
        self.samples = int(round(self.duration * self.sample_rate))
        self.seed = int(cfg["train"]["seed"])
        self.epoch_size = int(data["samples_per_epoch"]) if training else len(self.records)
        self.bad = BadFileRegistry(data["runtime_bad_files"])
        self.augmenter = (
            WaveformAugmenter(data["augmentation"], self.sample_rate, self.seed)
            if training
            else None
        )
        self.rng = np.random.default_rng(self.seed)

    def __len__(self) -> int:
        return self.epoch_size

    def _rng(self, index: int) -> np.random.Generator:
        worker = get_worker_info()
        if self.training:
            return self.rng
        worker_seed = 0 if worker is None else worker.id
        return np.random.default_rng(self.seed + index * 1009 + worker_seed)

    def __getitem__(self, index: int) -> dict:
        rng = self._rng(index)
        attempts = 0
        while attempts < len(self.records):
            record_index = (
                int(rng.integers(len(self.records))) if self.training else (index + attempts) % len(self.records)
            )
            record = self.records[record_index]
            attempts += 1
            if self.bad.contains(record.path):
                continue
            start_max = max(0.0, record.duration - self.duration)
            start = float(rng.uniform(0, start_max)) if self.training else start_max * 0.5
            try:
                audio = load_audio(
                    self.root / record.path,
                    sample_rate=self.sample_rate,
                    start=start,
                    duration=self.duration,
                    pad=True,
                )
                if len(audio) != self.samples or not np.isfinite(audio).all():
                    raise ValueError("decoded audio has invalid length or non-finite samples")
            except Exception as exc:
                self.bad.add(record.path, exc)
                continue
            augmentation = {}
            if self.augmenter is not None and rng.random() >= self.cfg["data"]["clean_probability"]:
                audio, augmentation = self.augmenter(audio)
            audio = quantile_normalize(audio, float(self.cfg["model"]["quantile_norm"]))
            return {
                "audio": torch.from_numpy(audio),
                "target": code_to_tokens(record.code),
                "code": record.code,
                "path": record.path,
                "start": start,
                "augmentation": json.dumps(augmentation, sort_keys=True),
            }
        raise RuntimeError("No readable catalogue tracks remain")
