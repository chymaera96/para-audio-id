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


@dataclass(frozen=True)
class AugmentationSchedule:
    clean_probability: float
    noise_probability: float
    rir_probability: float
    noise_rir_probability: float
    consistency_weight: float
    snr_bin_probabilities: tuple[float, float, float, float] | None
    phase: str
    rir_severity_quantile: float | None = None

    @property
    def background_noise_probability(self) -> float:
        return self.noise_probability + self.noise_rir_probability

    @property
    def room_ir_probability(self) -> float:
        return self.rir_probability + self.noise_rir_probability


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


def tc9_noise_consistency_schedule(step: int) -> NoiseConsistencySchedule:
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


def tc11_augmentation_schedule(step: int) -> AugmentationSchedule:
    """Fixed 175K clean/noise/RIR curriculum for tc11."""
    if step < 0:
        raise ValueError("Global step cannot be negative")
    snr_bins = (0.40, 0.30, 0.20, 0.10)
    if step < 50_000:
        return AugmentationSchedule(1.0, 0.0, 0.0, 0.0, 0.0, None, "clean")
    if step < 62_500:
        progress = (step - 50_000) / 12_500
        noise = 0.75 * progress
        return AugmentationSchedule(
            1.0 - noise,
            noise,
            0.0,
            0.0,
            0.10 * progress,
            snr_bins,
            "noise_ramp",
        )
    if step < 87_500:
        return AugmentationSchedule(
            0.25, 0.75, 0.0, 0.0, 0.10, snr_bins, "noise_steady"
        )
    if step < 100_000:
        progress = (step - 87_500) / 12_500
        return AugmentationSchedule(
            0.25,
            0.75 - 0.20 * progress,
            0.20 * progress,
            0.0,
            0.10,
            snr_bins,
            "rir_ramp",
        )
    if step < 112_500:
        progress = (step - 100_000) / 12_500
        return AugmentationSchedule(
            0.25,
            0.55 - 0.20 * progress,
            0.20,
            0.20 * progress,
            0.10,
            snr_bins,
            "combined_ramp",
        )
    return AugmentationSchedule(
        0.25, 0.35, 0.20, 0.20, 0.10, snr_bins, "consolidation"
    )


def resolved_augmentation_schedule(
    step: int, schedule: dict
) -> AugmentationSchedule:
    """Resolve the selected robustness curriculum at an optimizer step."""
    if step < 0:
        raise ValueError("Global step cannot be negative")
    name = schedule.get("name")
    if schedule.get("curriculum") == "tc12_noise_rir_curriculum_v1":
        clean_end = int(schedule["clean_until_step"])
        degradation_end = int(schedule["degradation_ramp_until_step"])
        combined_end = int(schedule["combined_ramp_until_step"])
        snr_bins = tuple(
            float(value) for value in schedule["snr_bin_probabilities"]
        )
        consistency = float(schedule["consistency_weight"])
        if step < clean_end:
            return AugmentationSchedule(
                1.0, 0.0, 0.0, 0.0, 0.0, None, "clean", None
            )
        if step < degradation_end:
            progress = (step - clean_end) / max(1, degradation_end - clean_end)
            return AugmentationSchedule(
                1.0 - 0.60 * progress,
                0.30 * progress,
                0.30 * progress,
                0.0,
                consistency * progress,
                snr_bins,
                "noise_rir_ramp",
                (1.0 + progress) / 3.0,
            )
        if step < combined_end:
            progress = (step - degradation_end) / max(
                1, combined_end - degradation_end
            )
            return AugmentationSchedule(
                0.40 - 0.30 * progress,
                0.30 + 0.05 * progress,
                0.30,
                0.25 * progress,
                consistency,
                snr_bins,
                "combined_ramp",
                (2.0 + progress) / 3.0,
            )
        return AugmentationSchedule(
            0.10,
            0.35,
            0.30,
            0.25,
            consistency,
            snr_bins,
            "full_distribution",
            1.0,
        )
    clean_end = int(schedule["clean_until_step"])
    ramp_end = int(schedule["noise_ramp_until_step"])
    snr_bins = tuple(float(value) for value in schedule["snr_bin_probabilities"])
    consistency = float(schedule["consistency_weight"])
    if step < clean_end:
        return AugmentationSchedule(1.0, 0.0, 0.0, 0.0, 0.0, None, "clean")
    if step < ramp_end:
        progress = (step - clean_end) / max(1, ramp_end - clean_end)
        noise = 0.75 * progress
        return AugmentationSchedule(
            1.0 - noise,
            noise,
            0.0,
            0.0,
            consistency * progress,
            snr_bins,
            "noise_ramp",
        )
    if name == "noise":
        return AugmentationSchedule(
            0.25, 0.75, 0.0, 0.0, consistency, snr_bins, "noise_steady"
        )
    if name != "noise-rir":
        raise ValueError(f"Unknown resolved schedule {name!r}")
    noise_end = int(schedule["noise_steady_until_step"])
    rir_end = int(schedule["rir_ramp_until_step"])
    combined_end = int(schedule["combined_ramp_until_step"])
    if step < noise_end:
        return AugmentationSchedule(
            0.25, 0.75, 0.0, 0.0, consistency, snr_bins, "noise_steady"
        )
    if step < rir_end:
        progress = (step - noise_end) / max(1, rir_end - noise_end)
        return AugmentationSchedule(
            0.25,
            0.75 - 0.20 * progress,
            0.20 * progress,
            0.0,
            consistency,
            snr_bins,
            "rir_ramp",
        )
    if step < combined_end:
        progress = (step - rir_end) / max(1, combined_end - rir_end)
        return AugmentationSchedule(
            0.25,
            0.55 - 0.20 * progress,
            0.20,
            0.20 * progress,
            consistency,
            snr_bins,
            "combined_ramp",
        )
    return AugmentationSchedule(
        0.25, 0.35, 0.20, 0.20, consistency, snr_bins, "consolidation"
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


def deterministic_augmentation_parameters(
    keys: list[int],
    *,
    schedule: AugmentationSchedule,
    seed: int,
    step: int,
    batch_idx: int,
) -> tuple[list[str], list[float], list[str | None]]:
    probabilities = (
        schedule.clean_probability,
        schedule.noise_probability,
        schedule.rir_probability,
        schedule.noise_rir_probability,
    )
    if any(value < 0.0 or value > 1.0 for value in probabilities):
        raise ValueError("Augmentation probabilities must be between zero and one")
    if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-9):
        raise ValueError("Augmentation probabilities must sum to one")
    snr_probabilities = schedule.snr_bin_probabilities
    if snr_probabilities is not None and not math.isclose(
        sum(snr_probabilities), 1.0, abs_tol=1e-9
    ):
        raise ValueError("SNR-bin probabilities must sum to one")
    categories = ("clean", "noise", "rir", "noise_rir")
    selected: list[str] = []
    snrs: list[float] = []
    bins: list[str | None] = []
    for pair, key in enumerate(keys):
        draw = stable_uniform(seed, step, batch_idx, pair, key, "category")
        cumulative = 0.0
        category = categories[-1]
        for name, probability in zip(categories, probabilities, strict=True):
            cumulative += probability
            if draw < cumulative:
                category = name
                break
        selected.append(category)
        if "noise" not in category:
            snrs.append(0.0)
            bins.append(None)
            continue
        if snr_probabilities is None:
            raise ValueError("Noise categories require SNR-bin probabilities")
        bin_draw = stable_uniform(seed, step, batch_idx, pair, key, "snr-bin")
        cumulative = 0.0
        bin_index = len(snr_probabilities) - 1
        for index, probability in enumerate(snr_probabilities):
            cumulative += probability
            if bin_draw < cumulative:
                bin_index = index
                break
        name, minimum, maximum = SNR_BINS[bin_index]
        bins.append(name)
        if name == "very_hard" and stable_uniform(
            seed, step, batch_idx, pair, key, "exact-zero"
        ) < 0.25:
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

    def load_training(self, key: object, *, samples: int | None = None) -> np.ndarray:
        return self._load(
            self.training_files,
            stable_uint64("train-noise", key),
            samples=samples,
        )

    def load_validation(
        self, key: object, *, samples: int | None = None
    ) -> np.ndarray:
        return self._load(
            self.validation_files,
            stable_uint64("validation-noise", key),
            samples=samples,
        )

    def _load(
        self, files: list[Path], seed: int, *, samples: int | None = None
    ) -> np.ndarray:
        target_samples = self.samples if samples is None else int(samples)
        if target_samples < 1:
            raise ValueError("Requested background-noise length must be positive")
        for attempt in range(len(files)):
            path = files[(seed + attempt) % len(files)]
            try:
                info = sf.info(path)
                source_samples = round(info.duration * self.sample_rate)
                if source_samples >= target_samples:
                    maximum_start = source_samples - target_samples
                    offset_samples = (
                        stable_uint64(seed, attempt, "offset") % (maximum_start + 1)
                    )
                    audio = load_audio(
                        path,
                        sample_rate=self.sample_rate,
                        start=offset_samples / self.sample_rate,
                        duration=target_samples / self.sample_rate,
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
                            audio, math.ceil(target_samples / len(audio))
                        )[:target_samples]
                if (
                    len(audio) == target_samples
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
