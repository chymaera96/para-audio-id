from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
from audiomentations import PitchShift, TimeStretch

from .audio import load_audio


def rms_normalize(audio: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return audio / max(float(np.sqrt(np.mean(np.square(audio)))), eps)


def peak_normalize(audio: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return audio / max(float(np.max(np.abs(audio))), eps)


def convolve_ir(audio: np.ndarray, ir: np.ndarray) -> np.ndarray:
    size = len(audio) + len(ir) - 1
    result = np.fft.irfft(np.fft.rfft(audio, n=size) * np.fft.rfft(ir, n=size), n=size)
    return peak_normalize(result[: len(audio)]).astype(np.float32)


class WaveformAugmenter:
    def __init__(self, cfg: dict, sample_rate: int, seed: int = 1337):
        self.cfg = cfg
        self.sample_rate = sample_rate
        self.rng = np.random.default_rng(seed)
        self.background = self._files(cfg["background"])
        self.room_ir = self._files(cfg["room_ir"])
        self.microphone_ir = self._files(cfg["microphone_ir"])
        self._validate_assets()

    @staticmethod
    def _files(cfg: dict) -> list[Path]:
        if not cfg.get("enabled", False):
            return []
        root = Path(cfg["root"])
        if not root.exists():
            return []
        suffixes = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
        return sorted(path for path in root.rglob("*") if path.suffix.lower() in suffixes)

    def _validate_assets(self) -> None:
        for name, files in (
            ("background", self.background),
            ("room_ir", self.room_ir),
            ("microphone_ir", self.microphone_ir),
        ):
            if self.cfg[name].get("enabled", False) and not files:
                raise FileNotFoundError(
                    f"augmentation.{name} is enabled but no audio exists under "
                    f"{self.cfg[name].get('root')!r}"
                )

    def _maybe(self, section: dict) -> bool:
        return section.get("enabled", False) and self.rng.random() < section.get("probability", 1)

    def _asset(self, files: list[Path], samples: int | None = None) -> np.ndarray:
        audio = load_audio(
            files[int(self.rng.integers(len(files)))],
            sample_rate=self.sample_rate,
            duration=None,
        )
        if samples is None:
            return audio
        if len(audio) < samples:
            audio = np.tile(audio, int(np.ceil(samples / max(1, len(audio)))))
        start = int(self.rng.integers(0, len(audio) - samples + 1))
        return audio[start : start + samples]

    def __call__(self, audio: np.ndarray) -> tuple[np.ndarray, dict]:
        output = np.asarray(audio, dtype=np.float32)
        applied: dict[str, float | bool] = {}
        pitch = self.cfg.get("pitch_shift", {})
        if self._maybe(pitch):
            semitones = float(self.rng.uniform(*pitch["semitones"]))
            output = PitchShift(
                min_semitones=semitones, max_semitones=semitones, p=1.0
            )(samples=output, sample_rate=self.sample_rate)
            applied["pitch_semitones"] = semitones
        stretch = self.cfg.get("time_stretch", {})
        if self._maybe(stretch):
            rate = float(self.rng.uniform(*stretch["rate"]))
            output = TimeStretch(min_rate=rate, max_rate=rate, p=1.0)(
                samples=output, sample_rate=self.sample_rate
            )
            applied["stretch_rate"] = rate
        resampling = self.cfg.get("resampling", {})
        if self._maybe(resampling):
            factor = float(self.rng.uniform(*resampling["factor"]))
            target_rate = max(1, int(round(self.sample_rate / factor)))
            output = librosa.resample(output, orig_sr=self.sample_rate, target_sr=target_rate)
            applied["playback_factor"] = factor
        background = self.cfg["background"]
        if self._maybe(background):
            noise = self._asset(self.background, len(output))
            snr = float(self.rng.uniform(*background["snr_db"]))
            output = (10 ** (snr / 20)) * rms_normalize(output) + rms_normalize(noise)
            output = peak_normalize(output)
            applied["background_snr_db"] = snr
        for name, files in (("room_ir", self.room_ir), ("microphone_ir", self.microphone_ir)):
            section = self.cfg[name]
            if self._maybe(section):
                output = convolve_ir(output, self._asset(files))
                applied[name] = True
        target = len(audio)
        if len(output) < target:
            output = np.pad(output, (0, target - len(output)))
        return np.asarray(output[:target], dtype=np.float32), applied
