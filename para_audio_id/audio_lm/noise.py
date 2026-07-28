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
