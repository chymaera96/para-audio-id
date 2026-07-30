from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict
from itertools import islice
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from ..audio import load_audio
from ..catalogue import CatalogueRecord
from .noise import (
    BackgroundNoiseAssets,
    deterministic_consistency_noise_parameters,
    mix_background_noise,
    stable_uint64,
    tc7_noise_consistency_schedule,
)


CROP_POLICY = "online_random_crop_24k_v2"
REPLACEMENT_POLICY = "deterministic_identity_replacement_v1"
TC6_MONITOR_GRIDS = {
    "canonical": (0.0, 5.0, 10.0, 15.0, 20.0, 25.0),
    "shifted": (
        1.0,
        2.0,
        3.0,
        4.0,
        6.0,
        7.0,
        8.0,
        9.0,
        11.0,
        12.0,
        13.0,
        14.0,
        16.0,
        17.0,
        18.0,
        19.0,
        21.0,
        22.0,
        23.0,
        24.0,
    ),
    "heldout": (2.5, 7.5, 12.5, 17.5, 22.5),
}


def random_start_sample(
    record: CatalogueRecord,
    *,
    sample_rate: int,
    crop_samples: int,
    seed: int,
    optimizer_step: int,
    batch_idx: int,
    pair_slot: int,
    role: int,
    attempt: int = 0,
    reserved_sample: int | set[int] | list[int] | tuple[int, ...] | None = None,
) -> int:
    source_samples = max(0, round(float(record.duration) * sample_rate))
    maximum = max(0, source_samples - crop_samples)
    if maximum == 0:
        return 0
    start = stable_uint64(
        seed,
        optimizer_step,
        batch_idx,
        pair_slot,
        record.track_id,
        role,
        attempt,
        "crop-start",
    ) % (maximum + 1)
    if reserved_sample is not None:
        reserved = (
            {reserved_sample}
            if isinstance(reserved_sample, int)
            else set(reserved_sample)
        )
        for _ in range(maximum + 1):
            if start not in reserved:
                break
            start = (start + 1) % (maximum + 1)
        else:
            raise ValueError(
                f"Every valid crop start is reserved for track {record.track_id}"
            )
    return int(start)


def make_tc6_evaluation_manifest(
    records: list[CatalogueRecord],
    *,
    sample_rate: int,
    crop_duration: float,
    seed: int,
) -> list[dict]:
    rows = []
    for track_offset, record in enumerate(records):
        for view_offset, (view_type, starts) in enumerate(
            TC6_MONITOR_GRIDS.items()
        ):
            start = starts[(track_offset + seed + view_offset) % len(starts)]
            rows.append(
                {
                    "track_id": record.track_id,
                    "code": record.code,
                    "source_path": record.path,
                    "source_duration": float(record.duration),
                    "start_sample": round(float(start) * sample_rate),
                    "start": float(start),
                    "crop_duration": float(crop_duration),
                    "view_type": view_type,
                }
            )
    return rows


class OnlineTrackDataset(Dataset):
    def __init__(self, records: list[CatalogueRecord]):
        if not records:
            raise ValueError("Online track dataset cannot be empty")
        if len({record.track_id for record in records}) != len(records):
            raise ValueError("Online track dataset contains duplicate identities")
        if len({record.code for record in records}) != len(records):
            raise ValueError("Online track dataset contains duplicate codes")
        self.records = list(records)
        self.track_ids = [record.track_id for record in records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int | tuple[int, int, int, int]) -> dict:
        if not isinstance(index, tuple):
            record = self.records[index]
            return {"record_index": index, "record": record}
        record_index, optimizer_step, batch_idx, pair_slot = index
        return {
            "record_index": int(record_index),
            "record": self.records[record_index],
            "optimizer_step": int(optimizer_step),
            "batch_idx": int(batch_idx),
            "pair_slot": int(pair_slot),
        }


class RandomEvaluationDataset(Dataset):
    def __init__(self, manifest: list[dict]):
        self.manifest = list(manifest)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict:
        return dict(self.manifest[index])


class RandomEvaluationCollator:
    def __init__(
        self,
        *,
        audio_root: str | Path,
        noise_assets: BackgroundNoiseAssets,
        sample_rate: int,
        seed: int,
    ):
        self.audio_root = Path(audio_root)
        self.noise_assets = noise_assets
        self.sample_rate = int(sample_rate)
        self.seed = int(seed)

    def __call__(self, examples: list[dict]) -> dict:
        clean = []
        noise = []
        for row in examples:
            waveform = load_audio(
                self.audio_root / row["source_path"],
                sample_rate=self.sample_rate,
                start=float(row["start"]),
                duration=float(row["crop_duration"]),
                pad=True,
            )
            expected = round(self.sample_rate * float(row["crop_duration"]))
            if (
                len(waveform) != expected
                or not np.isfinite(waveform).all()
                or float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
                <= 1e-8
            ):
                raise ValueError(
                    f"Fixed tc6 evaluation crop is invalid: {row['source_path']}"
                )
            clean.append(waveform)
            noise.append(
                self.noise_assets.load_validation(
                    stable_uint64(
                        self.seed,
                        row["track_id"],
                        row["start_sample"],
                        "fixed-evaluation-noise",
                    )
                )
            )
        return {
            "clean_waveforms": torch.from_numpy(np.stack(clean)),
            "noise_waveforms": torch.from_numpy(np.stack(noise)),
            "track_id": [row["track_id"] for row in examples],
            "code": [row["code"] for row in examples],
            "start_sample": [row["start_sample"] for row in examples],
            "start": [row["start"] for row in examples],
            "view_type": [row["view_type"] for row in examples],
        }


class OnlineTrackBatchSampler(Sampler[list[tuple[int, int, int, int]]]):
    def __init__(
        self,
        dataset: OnlineTrackDataset,
        *,
        tracks_per_microbatch: int,
        accumulation_steps: int,
        seed: int,
        catalogue_pass: int = 0,
    ):
        if tracks_per_microbatch < 1 or accumulation_steps < 1:
            raise ValueError("Batch and accumulation sizes must be positive")
        self.dataset = dataset
        self.tracks_per_microbatch = int(tracks_per_microbatch)
        self.accumulation_steps = int(accumulation_steps)
        self.seed = int(seed)
        self.catalogue_pass = int(catalogue_pass)

    def set_epoch(self, catalogue_pass: int) -> None:
        self.catalogue_pass = int(catalogue_pass)

    def __len__(self) -> int:
        batches = math.ceil(len(self.dataset) / self.tracks_per_microbatch)
        return math.ceil(batches / self.accumulation_steps) * self.accumulation_steps

    def __iter__(self) -> Iterator[list[tuple[int, int, int, int]]]:
        indices = np.arange(len(self.dataset), dtype=np.int64)
        rng = np.random.default_rng(self.seed + self.catalogue_pass)
        rng.shuffle(indices)
        missing = (-len(indices)) % self.tracks_per_microbatch
        if missing:
            indices = np.concatenate((indices, indices[:missing]))
        first_batches: list[np.ndarray] = []
        yielded = 0
        for offset in range(0, len(indices), self.tracks_per_microbatch):
            selected = indices[offset : offset + self.tracks_per_microbatch]
            if len(first_batches) < self.accumulation_steps:
                first_batches.append(selected.copy())
            optimizer_step = (
                self.catalogue_pass * len(self) + yielded
            ) // self.accumulation_steps
            yield [
                (int(index), optimizer_step, yielded, slot)
                for slot, index in enumerate(selected)
            ]
            yielded += 1
        for padding_index in range(len(self) - yielded):
            batch_idx = yielded + padding_index
            selected = first_batches[padding_index % len(first_batches)]
            optimizer_step = (
                self.catalogue_pass * len(self) + batch_idx
            ) // self.accumulation_steps
            yield [
                (int(index), optimizer_step, batch_idx, slot)
                for slot, index in enumerate(selected)
            ]


class RandomCropCollator:
    def __init__(
        self,
        *,
        records: list[CatalogueRecord],
        audio_root: str | Path,
        noise_assets: BackgroundNoiseAssets,
        sample_rate: int,
        crop_duration: float,
        seed: int,
        reserved_starts: dict[str, set[int]],
        crop_retries: int = 4,
        replacement_retries: int = 32,
    ):
        self.records = records
        self.audio_root = Path(audio_root)
        self.noise_assets = noise_assets
        self.sample_rate = int(sample_rate)
        self.crop_duration = float(crop_duration)
        self.crop_samples = round(self.sample_rate * self.crop_duration)
        self.seed = int(seed)
        self.reserved_starts = dict(reserved_starts)
        self.crop_retries = int(crop_retries)
        self.replacement_retries = int(replacement_retries)

    def _load_role(
        self,
        record: CatalogueRecord,
        *,
        optimizer_step: int,
        batch_idx: int,
        pair_slot: int,
        role: int,
        avoid_start: int | None = None,
    ) -> tuple[np.ndarray, int, int]:
        last_error: BaseException | None = None
        for attempt in range(self.crop_retries):
            start = random_start_sample(
                record,
                sample_rate=self.sample_rate,
                crop_samples=self.crop_samples,
                seed=self.seed,
                optimizer_step=optimizer_step,
                batch_idx=batch_idx,
                pair_slot=pair_slot,
                role=role,
                attempt=attempt,
                reserved_sample=self.reserved_starts.get(record.track_id),
            )
            maximum = max(
                0, round(record.duration * self.sample_rate) - self.crop_samples
            )
            if avoid_start is not None and maximum and start == avoid_start:
                start = (start + 1) % (maximum + 1)
            try:
                waveform = load_audio(
                    self.audio_root / record.path,
                    sample_rate=self.sample_rate,
                    start=start / self.sample_rate,
                    duration=self.crop_duration,
                    pad=True,
                )
                rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
                if (
                    len(waveform) != self.crop_samples
                    or not np.isfinite(waveform).all()
                    or rms <= 1e-8
                ):
                    raise ValueError("decoded crop is invalid or silent")
                return np.asarray(waveform, dtype=np.float32), start, attempt
            except Exception as exc:
                last_error = exc
        raise RuntimeError(
            f"{record.path} failed {self.crop_retries} deterministic crop attempts: "
            f"{last_error}"
        )

    def _replacement_candidates(
        self, original_index: int, *, optimizer_step: int, batch_idx: int, slot: int
    ) -> Iterator[int]:
        count = len(self.records)
        start = stable_uint64(
            self.seed, optimizer_step, batch_idx, slot, original_index, "replacement"
        ) % count
        for offset in range(count):
            yield int((start + offset) % count)

    def __call__(self, examples: list[dict]) -> dict:
        if not examples:
            raise ValueError("Random-crop batch cannot be empty")
        optimizer_steps = {row["optimizer_step"] for row in examples}
        batch_indices = {row["batch_idx"] for row in examples}
        if len(optimizer_steps) != 1 or len(batch_indices) != 1:
            raise ValueError("Random-crop batch has inconsistent sampler progress")
        optimizer_step = optimizer_steps.pop()
        batch_idx = batch_indices.pop()
        schedule = tc7_noise_consistency_schedule(optimizer_step)
        keys = [
            stable_uint64(
                self.seed,
                optimizer_step,
                batch_idx,
                row["pair_slot"],
                row["record"].track_id,
                "pair",
            )
            for row in examples
        ]
        selected, snrs, snr_bins = deterministic_consistency_noise_parameters(
            keys,
            schedule=schedule,
            seed=self.seed,
            step=optimizer_step,
            batch_idx=batch_idx,
        )
        decode_started = time.perf_counter()
        used_track_ids: set[str] = set()
        pairs = []
        failures = []
        replacements = []
        for row, noisy, snr, snr_bin in zip(
            examples, selected, snrs, snr_bins, strict=True
        ):
            original_index = int(row["record_index"])
            candidate_indices = [original_index]
            candidate_indices.extend(
                islice(
                    (
                        index
                        for index in self._replacement_candidates(
                            original_index,
                            optimizer_step=optimizer_step,
                            batch_idx=batch_idx,
                            slot=int(row["pair_slot"]),
                        )
                        if index != original_index
                    ),
                    self.replacement_retries,
                )
            )
            accepted = None
            for replacement_attempt, candidate_index in enumerate(
                candidate_indices[: self.replacement_retries + 1]
            ):
                record = self.records[candidate_index]
                if record.track_id in used_track_ids:
                    continue
                try:
                    first, first_start, first_attempts = self._load_role(
                        record,
                        optimizer_step=optimizer_step,
                        batch_idx=batch_idx,
                        pair_slot=int(row["pair_slot"]),
                        role=0,
                    )
                    if noisy:
                        second = first.copy()
                        second_start = first_start
                        second_attempts = first_attempts
                    else:
                        second, second_start, second_attempts = self._load_role(
                            record,
                            optimizer_step=optimizer_step,
                            batch_idx=batch_idx,
                            pair_slot=int(row["pair_slot"]),
                            role=1,
                            avoid_start=first_start,
                        )
                    accepted = (
                        record,
                        first,
                        second,
                        first_start,
                        second_start,
                        first_attempts + second_attempts,
                    )
                    if candidate_index != original_index:
                        replacements.append(
                            {
                                "original_track_id": row["record"].track_id,
                                "replacement_track_id": record.track_id,
                                "pair_slot": int(row["pair_slot"]),
                                "replacement_attempt": replacement_attempt,
                            }
                        )
                    break
                except Exception as exc:
                    failures.append(
                        {
                            "track_id": record.track_id,
                            "source_path": record.path,
                            "pair_slot": int(row["pair_slot"]),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
            if accepted is None:
                raise RuntimeError(
                    f"No valid deterministic replacement for pair slot "
                    f"{row['pair_slot']}"
                )
            record, first, second, first_start, second_start, retry_count = accepted
            used_track_ids.add(record.track_id)
            pairs.append(
                {
                    "record": record,
                    "first": first,
                    "second": second,
                    "first_start": first_start,
                    "second_start": second_start,
                    "is_noisy": bool(noisy),
                    "snr_db": float(snr),
                    "snr_bin": snr_bin,
                    "retry_count": retry_count,
                }
            )
        decode_seconds = time.perf_counter() - decode_started
        augmentation_started = time.perf_counter()
        noisy_pairs = [index for index, pair in enumerate(pairs) if pair["is_noisy"]]
        if noisy_pairs:
            signals = torch.from_numpy(
                np.stack([pairs[index]["first"] for index in noisy_pairs])
            )
            noises = torch.from_numpy(
                np.stack(
                    [
                        self.noise_assets.load_training(
                            stable_uint64(
                                self.seed,
                                optimizer_step,
                                batch_idx,
                                index,
                                pairs[index]["record"].track_id,
                                "noise-file",
                            )
                        )
                        for index in noisy_pairs
                    ]
                )
            )
            requested = torch.tensor(
                [pairs[index]["snr_db"] for index in noisy_pairs],
                dtype=torch.float32,
            )
            mixed, valid = mix_background_noise(signals, noises, requested)
            if not valid.all():
                raise RuntimeError("Validated random crop produced invalid noise mixing")
            for row_index, pair_index in enumerate(noisy_pairs):
                pairs[pair_index]["second"] = mixed[row_index].numpy()
        augmentation_seconds = time.perf_counter() - augmentation_started
        waveforms = []
        metadata = []
        for pair in pairs:
            record = pair["record"]
            for role, waveform, start, is_noisy in (
                ("anchor", pair["first"], pair["first_start"], False),
                (
                    "secondary",
                    pair["second"],
                    pair["second_start"],
                    pair["is_noisy"],
                ),
            ):
                waveforms.append(waveform)
                metadata.append(
                    {
                        "track_id": record.track_id,
                        "code": record.code,
                        "source_path": record.path,
                        "source_duration": float(record.duration),
                        "segment_start": start / self.sample_rate,
                        "start_sample": int(start),
                        "segment_duration": self.crop_duration,
                        "pair_role": role,
                        "is_noisy": bool(is_noisy),
                    }
                )
        return {
            "waveforms": torch.from_numpy(np.stack(waveforms)),
            "metadata": metadata,
            "planned_optimizer_step": int(optimizer_step),
            "planned_batch_idx": int(batch_idx),
            "schedule": asdict(schedule),
            "snr_bins": [
                pair["snr_bin"] for pair in pairs if pair["is_noisy"]
            ],
            "snrs": [pair["snr_db"] for pair in pairs if pair["is_noisy"]],
            "failed_crop_attempts": failures,
            "skipped_documents": 0,
            "replacements": replacements,
            "retry_count": sum(pair["retry_count"] for pair in pairs),
            "decode_seconds": decode_seconds,
            "augmentation_seconds": augmentation_seconds,
        }
