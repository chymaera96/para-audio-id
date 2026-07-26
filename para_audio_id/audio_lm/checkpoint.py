from __future__ import annotations

from pathlib import Path

import torch

from .model import AudioCausalLM
from .vocabulary import AudioLMVocabulary

ARCHITECTURE = "audio_lm_v1"


def validate_checkpoint_metadata(
    checkpoint: dict,
    *,
    tokenizer_fingerprint: str | None = None,
) -> None:
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError(
            "Checkpoint is not compatible with the audio_lm_v1 architecture"
        )
    required = {
        "tokenizer_spec",
        "tokenizer_fingerprint",
        "vocabulary",
        "model_config",
        "code_mapping_fingerprint",
        "validation_probe",
    }
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise ValueError(f"Audio-LM checkpoint is missing metadata: {missing}")
    if (
        tokenizer_fingerprint is not None
        and checkpoint["tokenizer_fingerprint"] != tokenizer_fingerprint
    ):
        raise ValueError("Checkpoint tokenizer fingerprint does not match")
    vocabulary = AudioLMVocabulary.from_dict(checkpoint["vocabulary"])
    vocabulary.validate()


def load_audio_lm(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[AudioCausalLM, AudioLMVocabulary, dict, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    validate_checkpoint_metadata(checkpoint)
    cfg = checkpoint["hyper_parameters"]
    vocabulary = AudioLMVocabulary.from_dict(checkpoint["vocabulary"])
    model = AudioCausalLM(cfg, vocabulary)
    state = {
        key.removeprefix("model."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, vocabulary, cfg, checkpoint
