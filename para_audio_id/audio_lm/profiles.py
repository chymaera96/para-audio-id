from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


SUPPORTED_DATABASE_SIZES = (10_000, 25_000, 50_000, 100_000)
DECODER_PROFILES = {
    "tiny": {"num_layers": 6, "hidden_size": 512, "num_attention_heads": 8},
    "small": {"num_layers": 12, "hidden_size": 768, "num_attention_heads": 12},
    "medium": {"num_layers": 24, "hidden_size": 1024, "num_attention_heads": 16},
}
SCHEDULE_NAMES = ("noise", "noise-rir")
NEW_TRAINING_PROTOCOL = "online_random_crop_consistency_profile_v2"
CAPACITY_TRAINING_PROTOCOL = "online_random_crop_clean_capacity_eight_codebook_v2"
LOSS_PROTOCOL = "tc5_family_weighted_consistency_v2"
CAPACITY_GLOBAL_TRACKS_PER_STEP = 80
CAPACITY_MAX_TRACKS_PER_DEVICE = 40
CAPACITY_SELECTED_CODEBOOKS = 8
CAPACITY_SEGMENT_DURATION = 2.0
CAPACITY_ID_DIGIT_WEIGHT = 32.0
CAPACITY_AUDIO_TARGETS = 400
CAPACITY_DOCUMENT_TOKENS = 408
CAPACITY_VOCABULARY_SIZE = 8_205


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
    legacy = isinstance(payload, list)
    if not legacy and not isinstance(payload, dict):
        raise ValueError("Training cohort must be a track-ID list or manifest object")
    track_ids = payload if legacy else payload.get("track_ids")
    metadata_invalid = not legacy and (
        payload.get("database_size", payload.get("count")) != expected_count
        or payload.get("count") != expected_count
    )
    if (
        metadata_invalid
        or not isinstance(track_ids, list)
        or len(track_ids) != expected_count
        or len(set(track_ids)) != expected_count
    ):
        raise ValueError(f"Training cohort must contain {expected_count} unique IDs")
    if not legacy and payload.get("catalogue_fingerprint") != catalogue_fingerprint(records):
        raise ValueError("Training cohort catalogue fingerprint does not match")
    by_track = {record.track_id: record for record in records}
    try:
        selected = [by_track[track_id] for track_id in track_ids]
    except KeyError as exc:
        raise ValueError(f"Training cohort identity is missing from catalogue: {exc}") from exc
    mapping = hashlib.sha256(
        "\n".join(f"{record.track_id}:{record.code}" for record in selected).encode()
    ).hexdigest()
    if len({record.code for record in selected}) != expected_count:
        raise ValueError("Training cohort catalogue codes are not unique")
    if not legacy and payload.get("code_mapping_fingerprint") != mapping:
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


def canonical_capacity_profile(
    *,
    database_size: int,
    decoder: str,
    target_exposures: int,
    tracks_per_optimizer_step: int,
    world_size: int = 1,
    tracks_per_device_microbatch: int | None = None,
    accumulate_grad_batches: int | None = None,
) -> dict[str, Any]:
    if target_exposures < 1 or tracks_per_optimizer_step < 1:
        raise ValueError("Capacity exposures and batch size must be positive")
    selections = database_size * target_exposures
    if selections % tracks_per_optimizer_step:
        raise ValueError(
            "Capacity exposure target must resolve to a whole optimizer-step count"
        )
    if tracks_per_device_microbatch is None or accumulate_grad_batches is None:
        if tracks_per_optimizer_step != CAPACITY_GLOBAL_TRACKS_PER_STEP:
            raise ValueError(
                "Explicit capacity parallelism is required for a non-standard "
                "global batch"
            )
        derived = capacity_parallelism(world_size)
        tracks_per_device_microbatch = int(
            derived["tracks_per_device_microbatch"]
        )
        accumulate_grad_batches = int(derived["accumulate_grad_batches"])
    if (
        tracks_per_device_microbatch
        * accumulate_grad_batches
        * world_size
        != tracks_per_optimizer_step
    ):
        raise ValueError("Capacity parallelism does not match the global batch")
    return {
        "version": 3,
        "experiment": "clean_capacity",
        "database_size": database_size,
        "training_tracks_manifest": cohort_manifest(database_size),
        "decoder": decoder_profile(decoder),
        "query": {
            "segment_duration_seconds": CAPACITY_SEGMENT_DURATION,
            "selected_codebooks": CAPACITY_SELECTED_CODEBOOKS,
            "audio_targets": CAPACITY_AUDIO_TARGETS,
            "document_tokens": CAPACITY_DOCUMENT_TOKENS,
            "vocabulary_size": CAPACITY_VOCABULARY_SIZE,
            "id_digit_weight": CAPACITY_ID_DIGIT_WEIGHT,
            "max_position_embeddings": 512,
        },
        "parallelism": {
            "world_size": world_size,
            "tracks_per_device_microbatch": tracks_per_device_microbatch,
            "accumulate_grad_batches": accumulate_grad_batches,
            "global_tracks_per_optimizer_step": tracks_per_optimizer_step,
        },
        "schedule": {
            "name": "clean",
            "protocol": CAPACITY_TRAINING_PROTOCOL,
            "loss_protocol": LOSS_PROTOCOL,
            "max_steps": selections // tracks_per_optimizer_step,
            "target_exposures": target_exposures,
            "tracks_per_optimizer_step": tracks_per_optimizer_step,
            "learning_rate_policy": "linear_warmup_then_constant_v1",
        },
    }


def capacity_parallelism(world_size: int) -> dict[str, int]:
    if world_size < 1 or CAPACITY_GLOBAL_TRACKS_PER_STEP % world_size:
        raise ValueError(
            "Capacity devices must be a positive divisor of the 80-track "
            "global optimizer batch"
        )
    tracks_per_device = min(
        CAPACITY_MAX_TRACKS_PER_DEVICE,
        CAPACITY_GLOBAL_TRACKS_PER_STEP // world_size,
    )
    denominator = tracks_per_device * world_size
    if CAPACITY_GLOBAL_TRACKS_PER_STEP % denominator:
        raise ValueError("Capacity parallelism cannot preserve the global batch")
    return {
        "world_size": world_size,
        "tracks_per_device_microbatch": tracks_per_device,
        "accumulate_grad_batches": CAPACITY_GLOBAL_TRACKS_PER_STEP // denominator,
        "global_tracks_per_optimizer_step": CAPACITY_GLOBAL_TRACKS_PER_STEP,
    }


def historical_checkpoint_profile(checkpoint: dict) -> dict[str, Any]:
    stored = checkpoint.get("resolved_training_profile")
    if stored is not None:
        if (
            stored.get("experiment") == "clean_capacity"
            and "parallelism" not in stored
            and stored.get("schedule", {}).get("protocol")
            == CAPACITY_TRAINING_PROTOCOL
        ):
            schedule = stored["schedule"]
            return canonical_capacity_profile(
                database_size=int(stored["database_size"]),
                decoder=stored["decoder"]["name"],
                target_exposures=int(schedule["target_exposures"]),
                tracks_per_optimizer_step=int(
                    schedule["tracks_per_optimizer_step"]
                ),
                world_size=1,
            )
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


def resolve_capacity_config(
    config: dict[str, Any],
    *,
    decoder: str | None = None,
    devices: int | None = None,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    cfg = deepcopy(config)
    data = cfg.setdefault("data", {})
    train = cfg.setdefault("train", {})
    trainer = cfg.setdefault("trainer", {})
    configured_size = int(data.get("database_size", 0))
    target_exposures = int(train.get("target_exposures", 560))
    resumed = checkpoint_training_profile(checkpoint) if checkpoint is not None else None
    if resumed is not None:
        if resumed.get("experiment") != "clean_capacity":
            raise ValueError("Capacity runs cannot resume a corruption-training checkpoint")
        if resumed.get("schedule", {}).get("protocol") != CAPACITY_TRAINING_PROTOCOL:
            raise ValueError(
                "Two-codebook capacity checkpoints cannot resume the "
                "eight-codebook capacity experiment"
            )
        if configured_size != int(resumed["database_size"]):
            raise ValueError("Configured database size does not match capacity checkpoint")
        resolved_decoder = resumed["decoder"]["name"] if decoder is None else decoder
        saved_parallelism = resumed.get("parallelism")
        if saved_parallelism is None:
            saved_parallelism = capacity_parallelism(1)
        saved_world_size = int(saved_parallelism["world_size"])
        if devices is not None and devices != saved_world_size:
            raise ValueError("Explicit device count does not match capacity checkpoint")
        parallelism = dict(saved_parallelism)
    else:
        resolved_decoder = decoder or "small"
        configured_devices = int(devices if devices is not None else trainer.get("devices", 1))
        parallelism = capacity_parallelism(configured_devices)
    tracks_per_step = int(parallelism["global_tracks_per_optimizer_step"])
    profile = canonical_capacity_profile(
        database_size=configured_size,
        decoder=resolved_decoder,
        target_exposures=target_exposures,
        tracks_per_optimizer_step=tracks_per_step,
        world_size=int(parallelism["world_size"]),
        tracks_per_device_microbatch=int(
            parallelism["tracks_per_device_microbatch"]
        ),
        accumulate_grad_batches=int(parallelism["accumulate_grad_batches"]),
    )
    if resumed is not None and profile != resumed:
        raise ValueError("Explicit capacity profile does not match resume checkpoint")
    data["max_training_tracks"] = configured_size
    data["training_tracks_manifest"] = profile["training_tracks_manifest"]
    model = cfg.setdefault("model", {})
    model.update(profile["decoder"])
    model.pop("name", None)
    model["max_position_embeddings"] = 512
    tokenizer = cfg.setdefault("tokenizer", {})
    tokenizer["selected_codebooks"] = CAPACITY_SELECTED_CODEBOOKS
    data["segment_duration"] = CAPACITY_SEGMENT_DURATION
    train["id_digit_weight"] = CAPACITY_ID_DIGIT_WEIGHT
    train["tracks_per_microbatch"] = int(
        parallelism["tracks_per_device_microbatch"]
    )
    trainer["accumulate_grad_batches"] = int(
        parallelism["accumulate_grad_batches"]
    )
    trainer["devices"] = int(parallelism["world_size"])
    trainer["strategy"] = "auto"
    train["max_steps"] = profile["schedule"]["max_steps"]
    train["schedule"] = {
        key: value for key, value in profile["schedule"].items() if key != "max_steps"
    }
    cfg["resolved_training_profile"] = profile
    return cfg
