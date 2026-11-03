"""CNN audio tokenizer with a fixed finite scalar quantization grid.

Token IDs are mixed-radix indices, least-significant coordinate first.
Feature normalization is fixed, and shared by training and frozen inference.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .tokenizer import TokenizerSpec, flatten_time_major
from .vocabulary import AudioLMVocabulary


class FSQ(nn.Module):
    def __init__(self, levels: list[int], groups: int = 1):
        super().__init__()
        if not levels or min(levels) < 2 or groups < 1:
            raise ValueError("FSQ needs levels >= 2 and at least one group")
        self.groups = groups
        self.dimensions = len(levels)
        self.codebook_size = math.prod(levels)
        self.register_buffer("levels", torch.tensor(levels, dtype=torch.long))
        self.register_buffer("basis", torch.tensor(
            [math.prod(levels[:i]) for i in range(len(levels))], dtype=torch.long))

    def forward(self, z):
        # Compute the quantization grid in FP32 even under mixed precision.
        z = z.float().reshape(*z.shape[:-1], self.groups, self.dimensions)
        levels = self.levels.to(z.dtype)
        half = (levels - 1) * (1 + 1e-3) / 2
        offset = torch.where(self.levels % 2 == 0, 0.5, 0.0)
        shift = torch.atanh(offset / half)
        bounded = torch.tanh(z + shift) * half - offset
        rounded = bounded.round()
        quantized = bounded + (rounded - bounded).detach()
        normalizer = torch.div(self.levels, 2, rounding_mode="floor")
        coordinates = quantized / normalizer
        digits = rounded.long() + normalizer
        ids = (digits * self.basis).sum(-1)
        return coordinates.flatten(-2), ids


class Residual(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels), nn.SiLU(),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, channels), nn.SiLU(),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation))

    def forward(self, x):
        return x + self.net(x)


class CNNFSQ(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        import torchaudio

        self.cfg = dict(cfg)
        self.sample_rate = int(cfg["sample_rate"])
        strides = cfg.get("temporal_strides", [2, 2])
        if len(strides) != 2 or any(s not in (1, 2) for s in strides):
            raise ValueError("temporal_strides must contain two strides, each 1 or 2")
        self.frame_rate = self.sample_rate / cfg["hop_length"] / math.prod(strides)
        self.quantizer = FSQ(cfg["levels"], cfg["groups"])
        c0, c1, c2 = cfg["channels"]
        width = len(cfg["levels"]) * cfg["groups"]
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate, n_fft=cfg["n_fft"],
            hop_length=cfg["hop_length"], n_mels=cfg["n_mels"],
            f_min=cfg["f_min"], f_max=cfg["f_max"], power=2,
            center=True, pad_mode="constant", norm="slaney", mel_scale="slaney")
        self.encoder = nn.Sequential(
            nn.Conv1d(cfg["n_mels"], c0, 5, padding=2), nn.SiLU(),
            nn.Conv1d(c0, c1, 5, stride=strides[0], padding=2), nn.SiLU(),
            nn.Conv1d(c1, c2, 5, stride=strides[1], padding=2),
            *[Residual(c2, d) for d in cfg["dilations"]],
            nn.GroupNorm(8, c2), nn.SiLU(), nn.Conv1d(c2, width, 1))
        self.decoder = nn.Sequential(
            nn.Conv1d(width, c2, 3, padding=1),
            Residual(c2, 1), Residual(c2, 2),
            nn.ConvTranspose1d(c2, c1, 4 if strides[1] == 2 else 3,
                               stride=strides[1], padding=1), nn.SiLU(),
            nn.ConvTranspose1d(c1, c0, 4 if strides[0] == 2 else 3,
                               stride=strides[0], padding=1), nn.SiLU(),
            nn.Conv1d(c0, cfg["n_mels"], 5, padding=2))

    def features(self, waveform):
        # Exactly ceil(samples/hop) frames: 200 for a two-second waveform.
        frames = math.ceil(waveform.shape[-1] / self.cfg["hop_length"])
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            mel = self.mel(waveform.float())[..., :frames]
            return (mel.clamp_min(self.cfg["log_floor"]).log10()
                    + self.cfg["log_offset"]) / self.cfg["log_scale"]

    def encode_features(self, mel):
        latent = self.encoder(mel).transpose(1, 2)
        coordinates, ids = self.quantizer(latent)
        return coordinates, ids.transpose(1, 2)

    def forward(self, mel):
        coordinates, ids = self.encode_features(mel)
        reconstruction = self.decoder(coordinates.transpose(1, 2))[..., :mel.shape[-1]]
        return reconstruction, coordinates, ids


class FrozenFSQTokenizer:
    """Same tokenize/spec/vocabulary interface as the frozen MuQ tokenizer."""
    def __init__(self, checkpoint: str | Path, device="cuda"):
        path = Path(checkpoint).resolve()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") != "cnn_fsq_tokenizer_v1":
            raise ValueError("Not a CNN FSQ tokenizer checkpoint")
        self.device = torch.device(device)
        self.model = CNNFSQ(payload["config"]["model"]).to(self.device).eval()
        self.model.load_state_dict(payload["model"])
        self.model.requires_grad_(False)
        self.sample_rate = self.model.sample_rate
        self.vocabulary = AudioLMVocabulary(
            self.model.quantizer.groups, self.model.quantizer.codebook_size)
        self.selected_codebooks = self.vocabulary.num_codebooks
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.spec = TokenizerSpec(
            architecture="cnn_fsq_tokenizer_v1", model_name=str(path),
            revision=digest, package_version="1", sample_rate=self.sample_rate,
            frame_rate=self.model.frame_rate,
            waveform_normalization="none_fixed_logmel_v1",
            num_available_codebooks=self.selected_codebooks,
            selected_codebooks=self.selected_codebooks,
            codebook_size=self.vocabulary.codebook_size,
            serialization="time_major_codebook_interleaved", preprocessing_version=1)

    @torch.inference_mode()
    def tokenize(self, waveform, **kwargs):
        # Lightning can wrap PSI training in BF16 autocast. Keep the frozen
        # quantization decisions identical to standalone FP32 evaluation.
        with torch.autocast(device_type=self.device.type, enabled=False):
            mel = self.model.features(waveform.to(self.device))
            _, ids = self.model.encode_features(mel)
        return flatten_time_major(ids, self.vocabulary)


def reconstruction_loss(clean_mel, reconstructed_clean, reconstructed_degraded):
    return 0.5 * (F.l1_loss(reconstructed_clean, clean_mel)
                  + F.l1_loss(reconstructed_degraded, clean_mel))
