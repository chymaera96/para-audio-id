from __future__ import annotations

from collections.abc import Iterator
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from ..audio import load_audio
from .noise import AUDIO_SUFFIXES, stable_uint64


def _room_identity(root: Path, path: Path) -> str:
    relative = path.relative_to(root)
    parts = relative.parts
    if len(parts) >= 3:
        return "/".join(parts[:2])
    if len(parts) == 2:
        return f"{parts[0]}/{path.stem}"
    return path.stem


class RoomImpulseResponseAssets:
    def __init__(
        self,
        training_root: str | Path,
        validation_root: str | Path,
        *,
        sample_rate: int,
    ):
        self.training_root = Path(training_root)
        self.validation_root = Path(validation_root)
        self.sample_rate = int(sample_rate)
        self.training_files = self._files(self.training_root)
        self.validation_files = self._files(self.validation_root)
        if not self.training_files:
            raise FileNotFoundError(
                f"No training room impulse responses found under {self.training_root}"
            )
        if not self.validation_files:
            raise FileNotFoundError(
                "No validation room impulse responses found under "
                f"{self.validation_root}"
            )
        training_rooms = {
            _room_identity(self.training_root, path) for path in self.training_files
        }
        validation_rooms = {
            _room_identity(self.validation_root, path)
            for path in self.validation_files
        }
        overlap = sorted(training_rooms & validation_rooms)
        if overlap:
            raise ValueError(
                f"Training and validation room sources overlap: {overlap[:5]}"
            )
        training_hashes = {self._content_hash(path) for path in self.training_files}
        validation_hashes = {
            self._content_hash(path) for path in self.validation_files
        }
        duplicate_content = training_hashes & validation_hashes
        if duplicate_content:
            raise ValueError(
                "Training and validation room-IR sets contain identical audio"
            )
        self.training_fingerprint = self._fingerprint(
            self.training_root, self.training_files
        )
        self.validation_fingerprint = self._fingerprint(
            self.validation_root, self.validation_files
        )
        self.training_files_by_severity = sorted(
            self.training_files,
            key=lambda path: (self._severity_score(path), path.as_posix()),
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
    def _content_hash(path: Path) -> str:
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        mono = np.asarray(audio.mean(axis=1), dtype="<f4")
        digest = hashlib.sha256()
        digest.update(str(sample_rate).encode())
        digest.update(str(mono.shape).encode())
        digest.update(np.ascontiguousarray(mono).tobytes())
        return digest.hexdigest()

    @staticmethod
    def _fingerprint(root: Path, files: list[Path]) -> str:
        rows = [
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": RoomImpulseResponseAssets._content_hash(path),
            }
            for path in files
        ]
        return hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @staticmethod
    def _severity_score(path: Path) -> float:
        """Estimate reverberation severity from post-peak energy-decay duration."""
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        mono = np.asarray(audio.mean(axis=1), dtype=np.float64)
        if not len(mono) or not np.isfinite(mono).all():
            return math.inf
        peak = int(np.argmax(np.abs(mono)))
        tail_energy = np.square(mono[peak:])
        total = float(tail_energy.sum())
        if total <= 1e-16:
            return math.inf
        end = int(np.searchsorted(np.cumsum(tail_energy), 0.99 * total))
        return end / float(sample_rate)

    def manifest(self) -> dict:
        return {
            "training_root": str(self.training_root),
            "validation_root": str(self.validation_root),
            "training_files": len(self.training_files),
            "validation_files": len(self.validation_files),
            "training_fingerprint": self.training_fingerprint,
            "validation_fingerprint": self.validation_fingerprint,
            "policy": "full_wet_two_second_past_context_severity_ranked_v2",
            "severity_measure": "post_peak_99_percent_energy_decay_seconds",
        }

    def load_training(
        self, key: object, *, severity_quantile: float = 1.0
    ) -> tuple[np.ndarray, str]:
        try:
            return next(
                self.iter_training(
                    key,
                    severity_quantile=severity_quantile,
                )
            )
        except StopIteration as exc:
            raise RuntimeError(
                "No readable non-silent room impulse responses remain"
            ) from exc

    def iter_training(
        self, key: object, *, severity_quantile: float = 1.0
    ) -> Iterator[tuple[np.ndarray, str]]:
        """Yield each eligible readable IR once in deterministic order."""
        if not 0.0 < severity_quantile <= 1.0:
            raise ValueError("RIR severity quantile must be in (0, 1]")
        eligible = max(
            1,
            math.ceil(len(self.training_files_by_severity) * severity_quantile),
        )
        yield from self._iter_load(
            self.training_root,
            self.training_files_by_severity[:eligible],
            stable_uint64("train-room-ir", key),
        )

    def load_validation(self, key: object) -> tuple[np.ndarray, str]:
        return self._load(
            self.validation_root,
            self.validation_files,
            stable_uint64("validation-room-ir", key),
        )

    def _load(
        self, root: Path, files: list[Path], seed: int
    ) -> tuple[np.ndarray, str]:
        try:
            return next(self._iter_load(root, files, seed))
        except StopIteration as exc:
            raise RuntimeError(
                "No readable non-silent room impulse responses remain"
            ) from exc

    def _iter_load(
        self, root: Path, files: list[Path], seed: int
    ) -> Iterator[tuple[np.ndarray, str]]:
        for attempt in range(len(files)):
            path = files[(seed + attempt) % len(files)]
            try:
                info = sf.info(path)
                if info.frames <= 0:
                    raise ValueError("empty room impulse response")
                ir = load_audio(
                    path,
                    sample_rate=self.sample_rate,
                    duration=None,
                )
                rms = float(np.sqrt(np.mean(np.square(ir, dtype=np.float64))))
                if not len(ir) or not np.isfinite(ir).all() or rms <= 1e-8:
                    raise ValueError("invalid or silent room impulse response")
                yield (
                    np.asarray(ir, dtype=np.float32),
                    path.relative_to(root).as_posix(),
                )
            except Exception:
                continue


def convolve_full_wet(
    audio_with_context: np.ndarray,
    ir: np.ndarray,
    *,
    past_context_samples: int,
    output_samples: int,
    eps: float = 1e-8,
) -> np.ndarray:
    audio = np.asarray(audio_with_context, dtype=np.float32)
    response = np.asarray(ir, dtype=np.float32)
    expected = past_context_samples + output_samples
    if audio.ndim != 1 or len(audio) != expected:
        raise ValueError(
            f"Context audio must contain exactly {expected} mono samples"
        )
    if response.ndim != 1 or not len(response):
        raise ValueError("Room impulse response must be non-empty and mono")
    size = len(audio) + len(response) - 1
    convolved = np.fft.irfft(
        np.fft.rfft(audio, n=size) * np.fft.rfft(response, n=size), n=size
    )[: len(audio)]
    peak = float(np.max(np.abs(convolved)))
    if not np.isfinite(peak) or peak <= eps:
        raise ValueError("Room convolution produced invalid or silent audio")
    normalized = convolved / peak
    query = normalized[past_context_samples:]
    if len(query) != output_samples or not np.isfinite(query).all():
        raise ValueError("Room convolution produced an invalid query")
    return np.asarray(query, dtype=np.float32)
