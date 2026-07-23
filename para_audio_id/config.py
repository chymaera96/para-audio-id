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


def with_overrides(config: dict[str, Any], *, run_id: str | None, wandb_online: bool) -> dict:
    cfg = deepcopy(config)
    wandb = cfg.setdefault("train", {}).setdefault("wandb", {})
    if wandb_online:
        wandb["enabled"] = True
        wandb["mode"] = "online"
    if run_id:
        cfg["train"]["run_id"] = run_id
        wandb["name"] = run_id
    return cfg
