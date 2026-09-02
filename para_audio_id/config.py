from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def with_overrides(
    config: dict[str, Any],
    *,
    run_id: str | None,
    wandb_online: bool,
    devices: int | None = None,
    database_size: int | None = None,
    decoder: str | None = None,
    schedule: str | None = None,
    selected_codebooks: int | None = None,
    distillation_weight: float | None = None,
    checkpoint: str | Path | None = None,
) -> dict:
    cfg = deepcopy(config)
    wandb = cfg.setdefault("train", {}).setdefault("wandb", {})
    if wandb_online:
        wandb["enabled"] = True
        wandb["mode"] = "online"
    if run_id:
        cfg["train"]["run_id"] = run_id
        wandb["name"] = run_id
    if devices is not None:
        if devices < 1:
            raise ValueError(f"devices must be positive, got {devices}")
        cfg.setdefault("trainer", {})["devices"] = devices
        cfg["trainer"]["strategy"] = "auto"
    if database_size is not None:
        cfg.setdefault("data", {})["database_size"] = database_size
        cfg["data"]["max_training_tracks"] = database_size
    from .audio_lm.profiles import resolve_training_config

    return resolve_training_config(
        cfg,
        decoder=decoder,
        schedule=schedule,
        selected_codebooks=selected_codebooks,
        distillation_weight=distillation_weight,
        checkpoint=checkpoint,
    )
