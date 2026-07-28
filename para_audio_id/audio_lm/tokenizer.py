from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import re
from typing import Any

import torch

from .vocabulary import AudioLMVocabulary


@dataclass(frozen=True)
class TokenizerSpec:
    architecture: str
    model_name: str
    revision: str
    package_version: str
    sample_rate: int
    frame_rate: float
    waveform_normalization: str
    num_available_codebooks: int
    selected_codebooks: int
    codebook_size: int
    serialization: str
    preprocessing_version: int

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def flatten_time_major(raw_codes: torch.Tensor, vocabulary: AudioLMVocabulary) -> torch.Tensor:
    if raw_codes.ndim != 3:
        raise ValueError(f"Expected [batch, codebook, frame], got {tuple(raw_codes.shape)}")
    if raw_codes.shape[1] != vocabulary.num_codebooks:
        raise ValueError(
            f"Expected {vocabulary.num_codebooks} selected codebooks, got {raw_codes.shape[1]}"
        )
    if raw_codes.numel() and (
        int(raw_codes.min()) < 0 or int(raw_codes.max()) >= vocabulary.codebook_size
    ):
        raise ValueError("RVQ code is outside the configured codebook range")
    offsets = torch.arange(
        vocabulary.num_codebooks, device=raw_codes.device, dtype=raw_codes.dtype
    ).view(1, -1, 1)
    separated = raw_codes + offsets * vocabulary.codebook_size
    return separated.transpose(1, 2).reshape(raw_codes.shape[0], -1)


def unflatten_time_major(
    tokens: torch.Tensor, frames: int, vocabulary: AudioLMVocabulary
) -> torch.Tensor:
    if tokens.ndim != 2 or tokens.shape[1] != frames * vocabulary.num_codebooks:
        raise ValueError("Token length is incompatible with frame and codebook counts")
    shaped = tokens.reshape(tokens.shape[0], frames, vocabulary.num_codebooks).transpose(1, 2)
    offsets = torch.arange(
        vocabulary.num_codebooks, device=tokens.device, dtype=tokens.dtype
    ).view(1, -1, 1)
    return shaped - offsets * vocabulary.codebook_size


class MuQRVQTokenizer:
    def __init__(
        self,
        model_name: str,
        *,
        revision: str = "main",
        selected_codebooks: int = 2,
        sample_rate: int = 24_000,
        device: str | torch.device = "cuda",
        lightweight: bool = False,
    ):
        try:
            from muq import MuQ
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise ImportError("The MuQ package is required for audio tokenization") from exc
        resolved_revision = (
            revision
            if re.fullmatch(r"[0-9a-f]{40}", revision)
            else HfApi().model_info(model_name, revision=revision).sha
        )
        kwargs: dict[str, Any] = {"revision": resolved_revision}
        loaded_model = MuQ.from_pretrained(model_name, **kwargs)
        self.device = torch.device(device)
        self.model_name = model_name
        self.revision = resolved_revision
        self.selected_codebooks = selected_codebooks
        self.sample_rate = sample_rate
        config = loaded_model.config
        if not bool(getattr(config, "use_rvq_target", False)):
            raise RuntimeError("Loaded MuQ checkpoint does not enable Mel-RVQ targets")
        inner = loaded_model.model
        rvq = getattr(inner, "rvq", None)
        if rvq is None or not hasattr(rvq, "n_codebooks"):
            raise RuntimeError("Loaded MuQ checkpoint has no accessible residual quantizer")
        self.available_codebooks = int(rvq.n_codebooks)
        self.codebook_size = int(rvq.codebook_size)
        if not 1 <= selected_codebooks <= self.available_codebooks:
            raise ValueError(
                f"Requested {selected_codebooks} of {self.available_codebooks} codebooks"
            )
        self.vocabulary = AudioLMVocabulary(selected_codebooks, self.codebook_size)
        self.vocabulary.validate()
        try:
            package_version = importlib.metadata.version("muq")
        except importlib.metadata.PackageNotFoundError:
            package_version = "unknown"
        self.spec = TokenizerSpec(
            architecture="muq_mel_rvq",
            model_name=model_name,
            revision=resolved_revision,
            package_version=package_version,
            sample_rate=sample_rate,
            frame_rate=float(getattr(config, "label_rate", 25)),
            waveform_normalization="none_before_muq_internal_preprocessing",
            num_available_codebooks=self.available_codebooks,
            selected_codebooks=selected_codebooks,
            codebook_size=self.codebook_size,
            serialization="time_major_codebook_interleaved",
            preprocessing_version=1,
        )
        self.lightweight = bool(lightweight)
        if self.lightweight:
            self.model = None
            self._frontend = inner.preprocessor_melspec_2048.to(self.device)
            self._feature_mean = torch.as_tensor(
                inner.stat["melspec_2048_mean"],
                device=self.device,
                dtype=torch.float32,
            )
            self._feature_std = torch.as_tensor(
                inner.stat["melspec_2048_std"],
                device=self.device,
                dtype=torch.float32,
            )
            if self._feature_mean.ndim == 1:
                self._feature_mean = self._feature_mean.view(1, -1, 1)
            if self._feature_std.ndim == 1:
                self._feature_std = self._feature_std.view(1, -1, 1)
            self._n_fold = int(inner.n_fold)
            self._rvq = rvq.to(self.device).eval()
            del inner
            del loaded_model
        else:
            self.model = loaded_model.to(self.device).eval()

    @torch.inference_mode()
    def raw_codes(self, waveform: torch.Tensor, *, verify_public_layout: bool = False) -> torch.Tensor:
        if waveform.ndim != 2:
            raise ValueError(f"Expected [batch, time] waveform, got {tuple(waveform.shape)}")
        waveform = waveform.to(self.device, dtype=torch.float32)
        if self.lightweight:
            mel = self._frontend(waveform.float())[..., :-1]
            mel = (mel - self._feature_mean) / self._feature_std
            batch, bands, samples = mel.shape
            if samples % self._n_fold:
                raise RuntimeError(
                    f"Mel frame count {samples} is not divisible by {self._n_fold}"
                )
            frames = samples // self._n_fold
            folded = (
                mel.reshape(batch, bands, frames, self._n_fold)
                .permute(0, 2, 3, 1)
                .reshape(batch, frames, self._n_fold * bands)
            )
            # MuQ's rearrange() produces [B, T, n_fold * mel_bands], but
            # ResidualVectorQuantize is Conv1d-based and get_rvq_codes() feeds it
            # [B, n_fold * mel_bands, T].
            rvq_input = folded.transpose(1, 2).contiguous()
            expected_channels = int(
                self._rvq.quantizers[0].in_proj.weight.shape[1]
            )
            if rvq_input.shape[1] != expected_channels:
                raise RuntimeError(
                    "Lightweight MuQ feature width does not match RVQ input channels: "
                    f"{rvq_input.shape[1]} != {expected_channels}"
                )
            result = self._rvq(rvq_input)
            if not isinstance(result, tuple) or len(result) < 2:
                raise RuntimeError("MuQ RVQ returned an unexpected output")
            direct = result[1].long()
            if direct.ndim != 3:
                raise RuntimeError(
                    f"Expected direct RVQ codes [B, Q, T], got {tuple(direct.shape)}"
                )
            return direct[:, : self.selected_codebooks, :]
        inner = self.model.model
        features = inner.preprocessing(waveform, features=inner.features)
        features = inner.normalize(features)
        features = inner.rearrange(features)
        mel = features["melspec_2048"]
        direct = inner.get_rvq_codes(mel.permute(0, 2, 1), None)
        if direct.ndim != 3:
            raise RuntimeError(f"Expected direct RVQ codes [B, Q, T], got {tuple(direct.shape)}")
        direct = direct.long()
        if verify_public_layout:
            public, _ = inner.get_targets(waveform)
            flattened = public["melspec_2048"].long()
            block_major = torch.cat(
                [direct[:, index, :] for index in range(direct.shape[1])], dim=-1
            )
            if not torch.equal(flattened, block_major):
                raise RuntimeError(
                    "MuQ get_targets output is not the expected block-major RVQ layout"
                )
        return direct[:, : self.selected_codebooks, :]

    @torch.inference_mode()
    def tokenize(
        self, waveform: torch.Tensor, *, verify_public_layout: bool = False
    ) -> torch.Tensor:
        return flatten_time_major(
            self.raw_codes(waveform, verify_public_layout=verify_public_layout),
            self.vocabulary,
        )

    @torch.inference_mode()
    def probe(self, waveform: torch.Tensor) -> dict:
        first = self.raw_codes(waveform, verify_public_layout=True)
        second = self.raw_codes(waveform, verify_public_layout=True)
        if not torch.equal(first, second):
            raise RuntimeError("MuQ RVQ tokenization is not deterministic in evaluation mode")
        tokens = flatten_time_major(first, self.vocabulary)
        restored = unflatten_time_major(tokens, first.shape[-1], self.vocabulary)
        if not torch.equal(first, restored):
            raise RuntimeError("Time-major RVQ serialization failed its round trip")
        return {
            "tokenizer": self.spec.to_dict(),
            "fingerprint": self.spec.fingerprint,
            "input_samples": int(waveform.shape[-1]),
            "batch_size": int(waveform.shape[0]),
            "raw_shape": list(first.shape),
            "frames": int(first.shape[-1]),
            "serialized_tokens_per_example": int(tokens.shape[-1]),
            "minimum_raw_code": int(first.min()),
            "maximum_raw_code": int(first.max()),
            "deterministic": True,
            "public_layout": "block_major_verified",
            "training_layout": "time_major_codebook_interleaved",
        }
