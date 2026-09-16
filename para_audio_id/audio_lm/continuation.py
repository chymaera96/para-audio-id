"""Prepare a separate, full-state scale continuation without editing its source."""
from copy import deepcopy
import hashlib
import math
from pathlib import Path

import torch

POLICY = "scale_continuation_ramp_hold_v1"


def continuation_multiplier(step, train_cfg):
    schedule = train_cfg["learning_rate_schedule"]
    elapsed = max(0, step - schedule["start_step"])
    fraction = min(elapsed / schedule["ramp_steps"], 1.0)
    lr = schedule["start_lr"] + fraction * (
        schedule["target_lr"] - schedule["start_lr"]
    )
    return lr / float(train_cfg["learning_rate"])


def fork_payload(payload, *, run_id, source, source_sha256,
                 additional_steps=25000, target_lr=5e-5, ramp_steps=2000):
    from .profiles import SCALE_VARIANT

    if additional_steps <= 0 or not 0 < ramp_steps <= additional_steps:
        raise ValueError("Require 0 < ramp_steps <= additional_steps")
    if not math.isfinite(target_lr) or target_lr <= 0:
        raise ValueError("Target LR must be positive and finite")
    if not run_id or Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("Use a simple new run ID without directories")
    result = deepcopy(payload)
    cfg = result["hyper_parameters"]
    profile = result["resolved_training_profile"]
    if profile.get("variant") != SCALE_VARIANT:
        raise ValueError("Continuation requires the scale medium 100K checkpoint")
    if profile.get("continuation"):
        raise ValueError("Resume the existing continuation instead of forking it again")
    if run_id == cfg["train"].get("run_id"):
        raise ValueError("Continuation must have a different run ID")
    step = int(result["global_step"])
    if step < int(profile["schedule"]["combined_ramp_until_step"]):
        raise ValueError("Source must have reached the final degradation mixture")
    optimizers = result["optimizer_states"]
    schedulers = result["lr_schedulers"]
    if len(optimizers) != 1 or len(schedulers) != 1:
        raise ValueError("Expected one AdamW optimizer and one scheduler")
    groups = optimizers[0]["param_groups"]
    start_lr = float(groups[0]["lr"])
    if any(float(group["lr"]) != start_lr for group in groups):
        raise ValueError("Optimizer groups have unequal LRs")
    if int(schedulers[0]["last_epoch"]) != step:
        raise ValueError("Scheduler and optimizer step disagree")
    schedule = {
        "policy": POLICY, "warmup_steps": int(cfg["train"]["warmup_steps"]),
        "start_step": step, "start_lr": start_lr,
        "target_lr": target_lr, "ramp_steps": ramp_steps,
    }
    provenance = {
        "source": str(source), "source_sha256": source_sha256,
        "source_step": step, "additional_steps": additional_steps,
        "run_id": run_id,
    }
    profile["continuation"] = provenance
    profile["schedule"]["max_steps"] = step + additional_steps
    profile["learning_rate_schedule"] = schedule
    cfg["resolved_training_profile"] = deepcopy(profile)
    cfg["train"]["max_steps"] = step + additional_steps
    cfg["train"]["learning_rate_schedule"] = deepcopy(schedule)
    cfg["train"]["run_id"] = run_id
    cfg["train"].setdefault("wandb", {})["name"] = run_id
    # A fresh checkpoint callback must never restore the source run's save paths.
    result["callbacks"] = {}
    return result


def prepare(source, run_id, **kwargs):
    from ..config import save_config

    source = Path(source).resolve()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    with source.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    result = fork_payload(payload, run_id=run_id, source=source,
                          source_sha256=digest, **kwargs)
    cfg = result["hyper_parameters"]
    directory = Path(cfg["train"]["checkpoint_dir"]) / run_id
    logs = Path(cfg["train"]["log_dir"]) / run_id
    if directory.resolve() == source.parent or directory.exists() or logs.exists():
        raise ValueError("New checkpoint and log directories must not already exist")
    directory.mkdir(parents=True)
    destination = directory / "continuation-start.ckpt"
    temporary = directory / "continuation-start.ckpt.tmp"
    torch.save(result, temporary)
    temporary.replace(destination)
    save_config(cfg, directory / "continuation.yaml")
    return destination, cfg
