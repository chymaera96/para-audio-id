from __future__ import annotations

import fcntl
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def quantile_normalize(audio: np.ndarray, q: float = 0.95, eps: float = 1e-8) -> np.ndarray:
    scale = max(float(np.quantile(np.abs(audio), q)), eps)
    return (audio / scale).astype(np.float32, copy=False)


def load_audio(
    path: str | Path,
    *,
    sample_rate: int,
    start: float = 0.0,
    duration: float | None = None,
    pad: bool = False,
) -> np.ndarray:
    path = Path(path)
    try:
        info = sf.info(path)
        offset = max(0, int(np.floor(start * info.samplerate)))
        frames = -1 if duration is None else int(np.ceil(duration * info.samplerate))
        audio, source_rate = sf.read(
            path, start=offset, frames=frames, dtype="float32", always_2d=False
        )
    except (sf.LibsndfileError, RuntimeError):
        audio, source_rate = librosa.load(
            path, sr=None, mono=True, offset=max(0.0, start), duration=duration
        )
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if source_rate != sample_rate:
        audio = librosa.resample(audio, orig_sr=source_rate, target_sr=sample_rate)
    if duration is not None:
        samples = int(round(duration * sample_rate))
        if len(audio) < samples and pad:
            audio = np.pad(audio, (0, samples - len(audio)))
        audio = audio[:samples]
    return np.asarray(audio, dtype=np.float32)


class BadFileRegistry:
    """A process-safe append-only registry used by all DataLoader workers and ranks."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._bad = self._read()
        self._mtime_ns = self.path.stat().st_mtime_ns if self.path.exists() else 0

    def _read(self) -> set[str]:
        if not self.path.exists():
            return set()
        with self.path.open("r", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            paths: set[str] = set()
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("path"):
                    paths.add(row["path"])
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            return paths

    def contains(self, relative_path: str) -> bool:
        if self.path.exists():
            mtime_ns = self.path.stat().st_mtime_ns
            if mtime_ns != self._mtime_ns:
                self._bad.update(self._read())
                self._mtime_ns = mtime_ns
        return relative_path in self._bad

    def add(self, relative_path: str, exc: BaseException) -> None:
        if relative_path in self._bad:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0)
            existing = {
                row["path"]
                for line in handle
                if line.strip() and (row := json.loads(line)).get("path")
            }
            if relative_path not in existing:
                handle.seek(0, 2)
                handle.write(
                    json.dumps(
                        {
                            "path": relative_path,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                handle.flush()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        self._bad.add(relative_path)
        self._mtime_ns = self.path.stat().st_mtime_ns
