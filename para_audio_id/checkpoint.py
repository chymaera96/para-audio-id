from __future__ import annotations

from pathlib import Path

import torch

from .model import ParametricAudioIdentifier


def load_network(
    checkpoint_path: str | Path, device: str | torch.device = "cpu"
) -> tuple[ParametricAudioIdentifier, dict, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint["hyper_parameters"]
    network = ParametricAudioIdentifier(cfg)
    state = {
        key.removeprefix("network."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("network.")
    }
    network.load_state_dict(state, strict=True)
    network.to(device).eval()
    return network, cfg, checkpoint
