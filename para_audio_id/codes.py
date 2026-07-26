from __future__ import annotations

import numpy as np


def assign_codes(n_tracks: int, seed: int = 1337) -> list[str]:
    if n_tracks != 100_000:
        raise ValueError("The complete five-digit code space requires exactly 100,000 tracks")
    values = np.arange(n_tracks)
    np.random.default_rng(seed).shuffle(values)
    return [f"{int(value):05d}" for value in values]
