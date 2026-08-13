from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


SUPPORTED_DATABASE_SIZES = (10_000, 25_000, 100_000)
DECODER_PROFILES = {
    "small": {"num_layers": 12, "hidden_size": 768, "num_attention_heads": 12},
    "medium": {"num_layers": 24, "hidden_size": 1024, "num_attention_heads": 16},
}
SCHEDULE_NAMES = ("noise", "noise-rir")
NEW_TRAINING_PROTOCOL = "online_random_crop_consistency_profile_v2"
LOSS_PROTOCOL = "tc5_family_weighted_consistency_v2"


def cohort_manifest(database_size: int) -> str:
    if database_size not in SUPPORTED_DATABASE_SIZES:
        raise ValueError(
            f"database_size must be one of {SUPPORTED_DATABASE_SIZES}, got {database_size}"
        )
    return f"data/training_tracks_{database_size // 1000}k.json"


def catalogue_fingerprint(records) -> str:
    payload = [
        {"track_id": record.track_id, "code": record.code, "path": record.path}
        for record in sorted(records, key=lambda record: record.track_id)
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_cohort_manifest(path: str | Path, records, expected_count: int) -> list[str]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"Missing {expected_count}-track cohort manifest: {source}. "
            "Run prepare_training_cohort.py first."
        )
    payload = json.loads(source.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Training cohort must use the fingerprinted manifest format")
    track_ids = payload.get("track_ids")
    if (
        payload.get("database_size", payload.get("count")) != expected_count
        or payload.get("count") != expected_count
        or not isinstance(track_ids, list)
        or len(track_ids) != expected_count
        or len(set(track_ids)) != expected_count
    ):
        raise ValueError(f"Training cohort must contain {expected_count} unique IDs")
    if payload.get("catalogue_fingerprint") != catalogue_fingerprint(records):
        raise ValueError("Training cohort catalogue fingerprint does not match")
    by_track = {record.track_id: record for record in records}
    try:
        selected = [by_track[track_id] for track_id in track_ids]
    except KeyError as exc:
        raise ValueError(f"Training cohort identity is missing from catalogue: {exc}") from exc
    mapping = hashlib.sha256(
        "\n".join(f"{record.track_id}:{record.code}" for record in selected).encode()
    ).hexdigest()
    if payload.get("code_mapping_fingerprint") != mapping:
        raise ValueError("Training cohort code mapping fingerprint does not match")
    return track_ids


def _scaled(value: int, database_size: int) -> int:
    return value * database_size // 10_000


def schedule_profile(name: str, database_size: int) -> dict[str, Any]:
    if name not in SCHEDULE_NAMES:
        raise ValueError(f"schedule must be one of {SCHEDULE_NAMES}, got {name!r}")
    total = _scaled(70_000, database_size)
    common = {
        "name": name,
        "protocol": NEW_TRAINING_PROTOCOL,
        "loss_protocol": LOSS_PROTOCOL,
        "max_steps": total,
        "consistency_weight": 0.10,
        "snr_bin_probabilities": [0.40, 0.30, 0.20, 0.10],
        "exact_zero_fraction_in_first_bin": 0.25,
    }
    if name == "noise":
        return {
            **common,
            "clean_until_step": _scaled(20_000, database_size),
            "noise_ramp_until_step": _scaled(25_000, database_size),
        }
    return {
        **common,
        "clean_until_step": _scaled(20_000, database_size),
        "noise_ramp_until_step": _scaled(25_000, database_size),
        "noise_steady_until_step": _scaled(35_000, database_size),
        "rir_ramp_until_step": _scaled(40_000, database_size),
        "combined_ramp_until_step": _scaled(45_000, database_size),
    }


def decoder_profile(name: str) -> dict[str, Any]:
    if name not in DECODER_PROFILES:
        raise ValueError(
            f"decoder must be one of {tuple(DECODER_PROFILES)}, got {name!r}"
        )
    return {"name": name, **DECODER_PROFILES[name]}


def canonical_training_profile(
    *, database_size: int, decoder: str, schedule: str
) -> dict[str, Any]:
    return {
        "version": 2,
        "database_size": database_size,
        "training_tracks_manifest": cohort_manifest(database_size),
        "decoder": decoder_profile(decoder),
        "schedule": schedule_profile(schedule, database_size),
    }


def historical_checkpoint_profile(checkpoint: dict) -> dict[str, Any]:
    stored = checkpoint.get("resolved_training_profile")
    if stored is not None:
        return stored
    track_count = len(checkpoint.get("training_track_ids", []))
    model = checkpoint.get("model_config") or checkpoint.get(
        "hyper_parameters", {}
    ).get("model", {})
    dimensions = (
        int(model.get("num_layers", 0)),
        int(model.get("hidden_size", 0)),
        int(model.get("num_attention_heads", 0)),
    )
    decoder = {
        (12, 768, 12): "small",
        (24, 1024, 16): "medium",
    }.get(dimensions)
    if decoder is None:
        raise ValueError(f"Checkpoint has an unknown decoder shape {dimensions}")
    protocol = checkpoint.get("training_protocol")
    if protocol == "online_random_crop_noise_rir_consistency_25k_v1":
        schedule = "noise-rir"
    elif protocol in {
        "token_budget_matched_two_second_noise_consistency_v1",
        "online_random_crop_noise_consistency_v1",
    }:
        schedule = "noise"
    else:
        raise ValueError(f"Checkpoint has an unsupported training protocol {protocol!r}")
    if track_count not in SUPPORTED_DATABASE_SIZES:
        raise ValueError(f"Checkpoint has unsupported catalogue size {track_count}")
    return canonical_training_profile(
        database_size=track_count, decoder=decoder, schedule=schedule
    )


def checkpoint_training_profile(path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return historical_checkpoint_profile(checkpoint)


def resolve_training_config(
    config: dict[str, Any],
    *,
    decoder: str | None = None,
    schedule: str | None = None,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    cfg = deepcopy(config)
    data = cfg.setdefault("data", {})
    database_size = int(data.get("database_size", data.get("max_training_tracks", 0)))
    resumed = checkpoint_training_profile(checkpoint) if checkpoint is not None else None
    if resumed is not None:
        database_size = int(resumed["database_size"])
        resolved_decoder = resumed["decoder"]["name"] if decoder is None else decoder
        resolved_schedule = resumed["schedule"]["name"] if schedule is None else schedule
    else:
        resolved_decoder = decoder or "small"
        resolved_schedule = schedule or "noise"
    profile = canonical_training_profile(
        database_size=database_size,
        decoder=resolved_decoder,
        schedule=resolved_schedule,
    )
    if resumed is not None and profile != resumed:
        raise ValueError("Explicit training profile does not match resume checkpoint")

    data["database_size"] = database_size
    data["max_training_tracks"] = database_size
    data["training_tracks_manifest"] = profile["training_tracks_manifest"]
    if resolved_schedule == "noise-rir":
        rir = data.get("room_ir")
        required = {"training_root", "validation_root", "past_context_duration"}
        if not isinstance(rir, dict) or required - rir.keys():
            missing = sorted(required - (rir.keys() if isinstance(rir, dict) else set()))
            raise ValueError(
                f"noise-rir requires complete data.room_ir configuration; missing {missing}"
            )
    model = cfg.setdefault("model", {})
    model.update(profile["decoder"])
    model.pop("name", None)
    train = cfg.setdefault("train", {})
    train["max_steps"] = profile["schedule"]["max_steps"]
    train["schedule"] = {
        key: value for key, value in profile["schedule"].items() if key != "max_steps"
    }
    cfg["resolved_training_profile"] = profile
    return cfg
