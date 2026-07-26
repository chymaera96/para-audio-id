from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from collections.abc import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .audio import BadFileRegistry, load_audio, quantile_normalize
from .augment import WaveformAugmenter
from .catalogue import CatalogueRecord, load_catalogue
from .codes import code_to_tokens

SEGMENT_POLICY_VERSION = 1


@dataclass(frozen=True)
class CanonicalSegment:
    record_index: int
    start: float


def canonical_starts(duration: float, query_duration: float, stride: float) -> tuple[float, ...]:
    if query_duration <= 0 or stride <= 0:
        raise ValueError("query_duration and segment_stride must be positive")
    if duration + 1e-9 < query_duration:
        raise ValueError(
            f"Recording duration {duration} is shorter than query duration {query_duration}"
        )
    last_start = max(0.0, duration - query_duration)
    count = int(math.floor(last_start / stride + 1e-9))
    starts = [round(index * stride, 9) for index in range(count + 1)]
    tail = round(last_start, 9)
    if not math.isclose(starts[-1], tail, abs_tol=1e-7):
        starts.append(tail)
    return tuple(starts)


def build_segment_inventory(
    records: list[CatalogueRecord], query_duration: float, stride: float
) -> tuple[list[CanonicalSegment], list[list[int]]]:
    segments: list[CanonicalSegment] = []
    by_record: list[list[int]] = []
    for record_index, record in enumerate(records):
        indices = []
        for start in canonical_starts(record.duration, query_duration, stride):
            indices.append(len(segments))
            segments.append(CanonicalSegment(record_index, start))
        by_record.append(indices)
    return segments, by_record


class CatalogueSegmentDataset(Dataset):
    def __init__(self, cfg: dict, *, training: bool):
        self.cfg = cfg
        self.training = training
        data = cfg["data"]
        self.root = Path(data["audio_root"])
        self.records = load_catalogue(data["catalogue"])
        self.sample_rate = int(cfg["model"]["sample_rate"])
        self.duration = float(data["query_duration"])
        self.stride = float(data["segment_stride"])
        self.samples = int(round(self.duration * self.sample_rate))
        self.seed = int(cfg["train"]["seed"])
        self.bad = BadFileRegistry(data["runtime_bad_files"])
        self.segments, self.segments_by_record = build_segment_inventory(
            self.records, self.duration, self.stride
        )
        augmentation_cfg = data["augmentation"]
        any_enabled = any(
            section.get("enabled", False) and float(section.get("probability", 0.0)) > 0
            for section in augmentation_cfg.values()
        )
        self.augmenter = (
            WaveformAugmenter(augmentation_cfg, self.sample_rate, self.seed)
            if training and any_enabled
            else None
        )
        self.rng = np.random.default_rng(self.seed)

    def __len__(self) -> int:
        return len(self.segments)

    def valid_record_indices(self) -> list[int]:
        bad_paths = self.bad.paths()
        return [
            index for index, record in enumerate(self.records) if record.path not in bad_paths
        ]

    def _replacement_segment(self, requested_index: int, attempt: int) -> CanonicalSegment:
        segment = self.segments[requested_index]
        replacement_record = (segment.record_index + attempt) % len(self.records)
        indices = self.segments_by_record[replacement_record]
        source_view = self.segments_by_record[segment.record_index].index(requested_index)
        return self.segments[indices[source_view % len(indices)]]

    def __getitem__(self, index: int) -> dict:
        requested = self.segments[index]
        segment = requested
        for attempt in range(len(self.records)):
            record = self.records[segment.record_index]
            if self.bad.contains(record.path):
                segment = self._replacement_segment(index, attempt + 1)
                continue
            try:
                audio = load_audio(
                    self.root / record.path,
                    sample_rate=self.sample_rate,
                    start=segment.start,
                    duration=self.duration,
                    pad=True,
                )
                if len(audio) != self.samples or not np.isfinite(audio).all():
                    raise ValueError("decoded audio has invalid length or non-finite samples")
            except Exception as exc:
                self.bad.add(record.path, exc)
                segment = self._replacement_segment(index, attempt + 1)
                continue
            augmentation = {}
            if (
                self.augmenter is not None
                and self.rng.random() >= float(self.cfg["data"]["clean_probability"])
            ):
                audio, augmentation = self.augmenter(audio)
            audio = quantile_normalize(audio, float(self.cfg["model"]["quantile_norm"]))
            return {
                "audio": torch.from_numpy(audio),
                "target": code_to_tokens(record.code),
                "code": record.code,
                "track_id": record.track_id,
                "path": record.path,
                "start": segment.start,
                "requested_track_id": self.records[requested.record_index].track_id,
                "augmentation": json.dumps(augmentation, sort_keys=True),
            }
        raise RuntimeError("No readable catalogue tracks remain")


class IdentityGroupedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: CatalogueSegmentDataset,
        *,
        songs_per_batch: int,
        views_per_song: int,
        seed: int,
        world_size: int = 1,
        rank: int = 0,
        exposure: int = 0,
    ):
        if songs_per_batch < 1 or views_per_song < 1:
            raise ValueError("songs_per_batch and views_per_song must be positive")
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError(f"Invalid distributed placement: rank={rank}, world_size={world_size}")
        if songs_per_batch % world_size:
            raise ValueError(
                f"Global songs_per_batch={songs_per_batch} must divide world_size={world_size}"
            )
        self.dataset = dataset
        self.songs_per_batch = songs_per_batch
        self.views_per_song = views_per_song
        self.seed = seed
        self.world_size = world_size
        self.rank = rank
        self.exposure = exposure

    def set_epoch(self, exposure: int) -> None:
        self.exposure = exposure

    def _ordered_records(self) -> list[int]:
        records = np.asarray(self.dataset.valid_record_indices(), dtype=np.int64)
        np.random.default_rng(self.seed + self.exposure).shuffle(records)
        return records.tolist()

    def __len__(self) -> int:
        return math.ceil(len(self.dataset.valid_record_indices()) / self.songs_per_batch)

    def __iter__(self) -> Iterator[list[int]]:
        records = self._ordered_records()
        if not records:
            raise RuntimeError("No readable catalogue tracks remain")
        missing = (-len(records)) % self.songs_per_batch
        if missing:
            records.extend(records[:missing])
        local_songs = self.songs_per_batch // self.world_size
        for offset in range(0, len(records), self.songs_per_batch):
            global_records = records[offset : offset + self.songs_per_batch]
            local_records = global_records[
                self.rank * local_songs : (self.rank + 1) * local_songs
            ]
            batch = []
            for record_index in local_records:
                candidates = self.dataset.segments_by_record[record_index]
                view_offset = self.exposure * self.views_per_song
                batch.extend(
                    candidates[(view_offset + view) % len(candidates)]
                    for view in range(self.views_per_song)
                )
            yield batch


# Backward-compatible import name for external callers.
CatalogueCropDataset = CatalogueSegmentDataset
