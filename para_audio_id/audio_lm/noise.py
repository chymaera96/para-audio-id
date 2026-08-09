from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from ..audio import load_audio

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


@dataclass(frozen=True)
class NoiseSchedule:
    probability: float
    snr_min_db: float | None
    snr_max_db: float | None


@dataclass(frozen=True)
class NoiseConsistencySchedule:
    probability: float
    consistency_weight: float
    snr_bin_probabilities: tuple[float, float, float, float] | None
    phase: str


def background_noise_schedule(step: int) -> NoiseSchedule:
    if step < 0:
        raise ValueError("Global step cannot be negative")
    if step < 20_000:
        return NoiseSchedule(0.0, None, None)
    if step < 25_000:
        progress = (step - 20_000) / 5_000
        return NoiseSchedule(0.25 * progress, 20.0, 30.0)
    if step < 35_000:
        progress = (step - 25_000) / 10_000
        return NoiseSchedule(0.25 + 0.25 * progress, 10.0, 30.0)
    if step < 45_000:
        progress = (step - 35_000) / 10_000
        return NoiseSchedule(0.50 + 0.25 * progress, 0.0, 30.0)
    return NoiseSchedule(0.75, 0.0, 30.0)


def _scaled_boundaries(max_steps: int) -> tuple[int, int, int, int, int]:
    if max_steps < 7:
        raise ValueError("max_steps is too small for the tc6 curriculum")
    return tuple(
        round(max_steps * reference / 70_000)
        for reference in (20_000, 30_000, 40_000, 55_000, 70_000)
    )


def noise_consistency_schedule(
    effective_step: int, *, max_steps: int = 70_000
) -> NoiseConsistencySchedule:
    if effective_step < 0:
        raise ValueError("Effective step cannot be negative")
    clean_end, easy_end, mixed_end, hard_end, final_end = _scaled_boundaries(
        max_steps
    )
    step = min(effective_step, final_end)
    if step < clean_end:
        return NoiseConsistencySchedule(0.0, 0.0, None, "clean")
    if step < easy_end:
        progress = (step - clean_end) / max(1, easy_end - clean_end)
        return NoiseConsistencySchedule(
            0.25 * progress,
            0.05 * progress,
            (0.00, 0.05, 0.25, 0.70),
            "easy",
        )
    if step < mixed_end:
        progress = (step - easy_end) / max(1, mixed_end - easy_end)
        return NoiseConsistencySchedule(
            0.25 + 0.25 * progress,
            0.05 + 0.05 * progress,
            (0.05, 0.15, 0.35, 0.45),
            "mixed",
        )
    if step < hard_end:
        progress = (step - mixed_end) / max(1, hard_end - mixed_end)
        return NoiseConsistencySchedule(
            0.50 + 0.25 * progress,
            0.10,
            (0.20, 0.30, 0.30, 0.20),
            "hardening",
        )
    return NoiseConsistencySchedule(
        0.75,
        0.10,
        (0.40, 0.30, 0.20, 0.10),
        "consolidation",
    )


def tc8_noise_consistency_schedule(step: int) -> NoiseConsistencySchedule:
    if step < 0:
        raise ValueError("Global step cannot be negative")
    if step < 20_000:
        return NoiseConsistencySchedule(0.0, 0.0, None, "clean")
    if step < 25_000:
        progress = (step - 20_000) / 5_000
        return NoiseConsistencySchedule(
            0.75 * progress,
            0.10 * progress,
            (0.40, 0.30, 0.20, 0.10),
            "ramp",
        )
    return NoiseConsistencySchedule(
        0.75,
        0.10,
        (0.40, 0.30, 0.20, 0.10),
        "steady",
    )


def stable_uint64(*values: object) -> int:
    payload = ":".join(str(value) for value in values).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def stable_uniform(*values: object) -> float:
    return stable_uint64(*values) / float(2**64)


def deterministic_noise_parameters(
    keys: list[int],
    *,
    probability: float,
    snr_min_db: float | None,
    snr_max_db: float | None,
    seed: int,
    step: int,
    batch_idx: int,
) -> tuple[list[bool], list[float]]:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Noise probability must be between zero and one")
    if (snr_min_db is None) != (snr_max_db is None):
        raise ValueError("SNR bounds must either both be set or both be disabled")
    selected = [
        stable_uniform(seed, step, pair, key, "apply") < probability
        for pair, key in enumerate(keys)
    ]
    snrs = []
    for pair, key in enumerate(keys):
        if snr_min_db is None or snr_max_db is None:
            snrs.append(0.0)
        else:
            uniform = stable_uniform(seed, step, pair, key, "snr")
            snrs.append(snr_min_db + uniform * (snr_max_db - snr_min_db))
    return selected, snrs


SNR_BINS = (
    ("very_hard", 0.0, 5.0),
    ("hard", 5.0, 10.0),
    ("medium", 10.0, 20.0),
    ("easy", 20.0, 30.0),
)


def deterministic_consistency_noise_parameters(
    keys: list[int],
    *,
    schedule: NoiseConsistencySchedule,
    seed: int,
    step: int,
    batch_idx: int,
) -> tuple[list[bool], list[float], list[str | None]]:
    if not 0.0 <= schedule.probability <= 1.0:
        raise ValueError("Noise probability must be between zero and one")
    probabilities = schedule.snr_bin_probabilities
    if probabilities is not None and not math.isclose(sum(probabilities), 1.0):
        raise ValueError("SNR-bin probabilities must sum to one")
    selected = []
    snrs = []
    bins: list[str | None] = []
    for pair, key in enumerate(keys):
        keep = (
            stable_uniform(seed, step, batch_idx, pair, key, "apply")
            < schedule.probability
        )
        selected.append(keep)
        if probabilities is None or not keep:
            snrs.append(0.0)
            bins.append(None)
            continue
        draw = stable_uniform(seed, step, batch_idx, pair, key, "snr-bin")
        cumulative = 0.0
        bin_index = len(probabilities) - 1
        for index, probability in enumerate(probabilities):
            cumulative += probability
            if draw < cumulative:
                bin_index = index
                break
        name, minimum, maximum = SNR_BINS[bin_index]
        bins.append(name)
        if (
            schedule.phase in {"consolidation", "ramp", "steady"}
            and name == "very_hard"
            and stable_uniform(seed, step, batch_idx, pair, key, "exact-zero")
            < 0.25
        ):
            snrs.append(0.0)
        else:
            position = stable_uniform(
                seed, step, batch_idx, pair, key, "snr-within-bin"
            )
            snrs.append(minimum + position * (maximum - minimum))
    return selected, snrs, bins


class BackgroundNoiseAssets:
    def __init__(
        self,
        training_root: str | Path,
        validation_root: str | Path,
        *,
        sample_rate: int,
        samples: int,
    ):
        self.training_root = Path(training_root)
        self.validation_root = Path(validation_root)
        self.sample_rate = int(sample_rate)
        self.samples = int(samples)
        self.training_files = self._files(self.training_root)
        self.validation_files = self._files(self.validation_root)
        if not self.training_files:
            raise FileNotFoundError(
                f"No training background noise found under {self.training_root}"
            )
        if not self.validation_files:
            raise FileNotFoundError(
                f"No validation background noise found under {self.validation_root}"
            )
        training_names = {path.name for path in self.training_files}
        validation_names = {path.name for path in self.validation_files}
        overlap = sorted(training_names & validation_names)
        if overlap:
            raise ValueError(
                "Training and validation background-noise sources overlap: "
                f"{overlap[:5]}"
            )
        self.training_fingerprint = self._fingerprint(
            self.training_root, self.training_files
        )
        self.validation_fingerprint = self._fingerprint(
            self.validation_root, self.validation_files
        )

    @staticmethod
    def _files(root: Path) -> list[Path]:
        if not root.exists():
            return []
        return sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
        )

    @staticmethod
    def _fingerprint(root: Path, files: list[Path]) -> str:
        rows = [
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
            }
            for path in files
        ]
        return hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def manifest(self) -> dict:
        return {
            "training_root": str(self.training_root),
            "validation_root": str(self.validation_root),
            "training_files": len(self.training_files),
            "validation_files": len(self.validation_files),
            "training_fingerprint": self.training_fingerprint,
            "validation_fingerprint": self.validation_fingerprint,
        }

    def load_training(self, key: object) -> np.ndarray:
        return self._load(self.training_files, stable_uint64("train-noise", key))

    def load_validation(self, key: object) -> np.ndarray:
        return self._load(
            self.validation_files, stable_uint64("validation-noise", key)
        )

    def _load(self, files: list[Path], seed: int) -> np.ndarray:
        for attempt in range(len(files)):
            path = files[(seed + attempt) % len(files)]
            try:
                info = sf.info(path)
                source_samples = round(info.duration * self.sample_rate)
                if source_samples >= self.samples:
                    maximum_start = source_samples - self.samples
                    offset_samples = (
                        stable_uint64(seed, attempt, "offset") % (maximum_start + 1)
                    )
                    audio = load_audio(
                        path,
                        sample_rate=self.sample_rate,
                        start=offset_samples / self.sample_rate,
                        duration=self.samples / self.sample_rate,
                        pad=False,
                    )
                else:
                    audio = load_audio(
                        path,
                        sample_rate=self.sample_rate,
                        duration=None,
                    )
                    if len(audio):
                        audio = np.tile(
                            audio, math.ceil(self.samples / len(audio))
                        )[: self.samples]
                if (
                    len(audio) == self.samples
                    and np.isfinite(audio).all()
                    and float(np.sqrt(np.mean(np.square(audio)))) > 1e-8
                ):
                    return np.asarray(audio, dtype=np.float32)
            except Exception:
                continue
        raise RuntimeError("No readable non-silent background-noise files remain")


def mix_background_noise(
    signal: torch.Tensor,
    noise: torch.Tensor,
    snr_db: torch.Tensor,
    *,
    eps: float = 1e-8,
    peak_limit: float = 0.999,
) -> tuple[torch.Tensor, torch.Tensor]:
    if signal.shape != noise.shape or signal.ndim != 2:
        raise ValueError("Signal and noise must have matching [batch, time] shapes")
    if snr_db.shape != (signal.shape[0],):
        raise ValueError("SNR must contain one value per waveform")
    signal_rms = signal.float().square().mean(dim=1).sqrt()
    noise_rms = noise.float().square().mean(dim=1).sqrt()
    valid = (signal_rms > eps) & (noise_rms > eps)
    scale = signal_rms / (
        noise_rms.clamp_min(eps) * torch.pow(10.0, snr_db.float() / 20.0)
    )
    mixed = signal.float() + noise.float() * scale[:, None]
    peak = mixed.abs().amax(dim=1)
    attenuation = torch.minimum(
        torch.ones_like(peak),
        torch.full_like(peak, peak_limit) / peak.clamp_min(eps),
    )
    mixed = mixed * attenuation[:, None]
    mixed = torch.where(valid[:, None], mixed, signal.float())
    return mixed, valid
