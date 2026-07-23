from __future__ import annotations

import numpy as np
import torch

BOS_TOKEN = 0
DIGIT_OFFSET = 1
VOCAB_SIZE = 11
CODE_LENGTH = 5


def assign_codes(n_tracks: int, seed: int = 1337) -> list[str]:
    if n_tracks != 100_000:
        raise ValueError("The complete five-digit code space requires exactly 100,000 tracks")
    values = np.arange(n_tracks)
    np.random.default_rng(seed).shuffle(values)
    return [f"{int(value):05d}" for value in values]


def code_to_tokens(code: str) -> torch.Tensor:
    if len(code) != CODE_LENGTH or not code.isdecimal():
        raise ValueError(f"Expected exactly five decimal digits, got {code!r}")
    return torch.tensor([int(char) + DIGIT_OFFSET for char in code], dtype=torch.long)


def tokens_to_code(tokens: torch.Tensor | list[int]) -> str:
    values = tokens.tolist() if isinstance(tokens, torch.Tensor) else list(tokens)
    if len(values) != CODE_LENGTH or any(value < 1 or value > 10 for value in values):
        raise ValueError(f"Expected five digit-token IDs in [1, 10], got {values}")
    return "".join(str(value - DIGIT_OFFSET) for value in values)


def teacher_forcing_inputs(targets: torch.Tensor) -> torch.Tensor:
    if targets.ndim != 2 or targets.shape[1] != CODE_LENGTH:
        raise ValueError(f"Expected [batch, 5] targets, got {tuple(targets.shape)}")
    bos = torch.full((targets.shape[0], 1), BOS_TOKEN, device=targets.device, dtype=torch.long)
    return torch.cat((bos, targets[:, :-1]), dim=1)
