from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


SUPPORTED_DATABASE_SIZES = (25_000,)
DECODER_PROFILES = {
    "small": {"num_layers": 12, "hidden_size": 768, "num_attention_heads": 12},
}
SCHEDULE_NAMES = ("noise-rir",)
SUPPORTED_SELECTED_CODEBOOKS = (2,)
TC13_ID_DIGIT_WEIGHT = 20.0
NEW_TRAINING_PROTOCOL = "tc13_five_second_task_anchored_simsiam_v1"
LOSS_PROTOCOL = "tc13_task_anchored_simsiam_v1"
TC12_CURRICULUM = "tc12_noise_rir_curriculum_v1"
TC13_LR_POLICY = "tc13_warmup_hold_linear_cosine_v1"


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
    if database_size != 25_000:
        raise ValueError("tc13 requires the 25K training cohort")
    common = {
        "name": name,
        "protocol": NEW_TRAINING_PROTOCOL,
        "loss_protocol": LOSS_PROTOCOL,
        "max_steps": 225_000,
        "predictive_weight": 0.10,
        "snr_bin_probabilities": [0.40, 0.30, 0.20, 0.10],
        "exact_zero_fraction_in_first_bin": 0.25,
    }
    return {
        **common,
        "curriculum": TC12_CURRICULUM,
        "clean_until_step": _scaled(4_000, database_size),
        "degradation_ramp_until_step": _scaled(12_000, database_size),
        "combined_ramp_until_step": _scaled(24_000, database_size),
    }


def decoder_profile(name: str) -> dict[str, Any]:
    if name not in DECODER_PROFILES:
        raise ValueError(
            f"decoder must be one of {tuple(DECODER_PROFILES)}, got {name!r}"
        )
    return {"name": name, **DECODER_PROFILES[name]}


def canonical_training_profile(
    *,
    database_size: int,
    decoder: str,
    schedule: str,
    selected_codebooks: int = 2,
) -> dict[str, Any]:
    if database_size != 25_000:
        raise ValueError("tc13 requires database_size=25000")
    if decoder != "small":
        raise ValueError("tc13 requires the small decoder")
    if schedule != "noise-rir":
        raise ValueError("tc13 requires the noise-rir schedule")
    if selected_codebooks != 2:
        raise ValueError("tc13 requires two MuQ codebooks")
    profile = {
        "version": 4,
        "variant": "tc13-task-anchored",
        "database_size": database_size,
        "training_tracks_manifest": cohort_manifest(database_size),
        "decoder": decoder_profile(decoder),
        "schedule": schedule_profile(schedule, database_size),
        "auxiliary": {
            "protocol": LOSS_PROTOCOL,
            "summary_weight": 0.10,
            "maximum_predictive_weight": 0.10,
            "predictive_weight_schedule": {
                "zero_until_step": 10_000,
                "ramp_until_step": 30_000,
                "maximum_weight": 0.10,
            },
            "summary_digits": 5,
            "digit_classes": 10,
            "projector": [768, 1024, 256],
            "predictor": [256, 1024, 256],
            "normalization": "layer_norm",
            "clean_target_detached_after_projector": True,
            "ema_target_encoder": False,
            "contrastive_negatives": False,
        },
    }
    profile["learning_rate_schedule"] = {
        "policy": TC13_LR_POLICY,
        "warmup_steps": 500,
        "hold_until_step": 60_000,
        "linear_decay_until_step": 140_000,
        "final_learning_rate_ratio": 0.05,
    }
    return profile


def historical_checkpoint_profile(checkpoint: dict) -> dict[str, Any]:
    stored = checkpoint.get("resolved_training_profile")
    if stored is None or stored.get("variant") != "tc13-task-anchored":
        raise ValueError(
            "Only task-anchored tc13 checkpoints can be resumed on this branch"
        )
    return stored


def checkpoint_training_profile(path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return historical_checkpoint_profile(checkpoint)


def _checkpoint_query_profile(checkpoint: dict[str, Any]) -> dict[str, Any] | None:
    tokenizer = checkpoint.get("tokenizer_spec") or checkpoint.get(
        "hyper_parameters", {}
    ).get("tokenizer", {})
    selected = tokenizer.get("selected_codebooks")
    if selected is None:
        return None
    selected = int(selected)
    query = checkpoint.get("query_spec", {})
    train = checkpoint.get("hyper_parameters", {}).get("train", {})
    weight = float(
        query.get(
            "id_digit_weight",
            train.get(
                "id_digit_weight", TC13_ID_DIGIT_WEIGHT
            ),
        )
    )
    return {
        "selected_codebooks": selected,
        "id_digit_weight": weight,
    }


def resolve_query_profile(selected_codebooks: int) -> dict[str, Any]:
    if selected_codebooks not in SUPPORTED_SELECTED_CODEBOOKS:
        raise ValueError(
            "selected codebooks must be one of "
            f"{SUPPORTED_SELECTED_CODEBOOKS}, got {selected_codebooks}"
        )
    return {
        "selected_codebooks": selected_codebooks,
        "id_digit_weight": TC13_ID_DIGIT_WEIGHT,
    }


def resolve_training_config(
    config: dict[str, Any],
    *,
    decoder: str | None = None,
    schedule: str | None = None,
    selected_codebooks: int | None = None,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    cfg = deepcopy(config)
    data = cfg.setdefault("data", {})
    database_size = int(data.get("database_size", data.get("max_training_tracks", 0)))
    checkpoint_payload = (
        torch.load(checkpoint, map_location="cpu", weights_only=False)
        if checkpoint is not None
        else None
    )
    resumed = (
        historical_checkpoint_profile(checkpoint_payload)
        if checkpoint_payload is not None
        else None
    )
    if resumed is not None:
        database_size = int(resumed["database_size"])
        resolved_decoder = resumed["decoder"]["name"] if decoder is None else decoder
        resolved_schedule = resumed["schedule"]["name"] if schedule is None else schedule
    else:
        resolved_decoder = decoder or "small"
        resolved_schedule = schedule or "noise-rir"
    if resumed is not None:
        if resolved_decoder != resumed["decoder"]["name"] or (
            resolved_schedule != resumed["schedule"]["name"]
        ):
            raise ValueError("Explicit training profile does not match resume checkpoint")
        profile = deepcopy(resumed)
    else:
        configured_codebooks = int(
            cfg.setdefault("tokenizer", {}).get("selected_codebooks", 2)
        )
        profile_codebooks = (
            selected_codebooks
            if selected_codebooks is not None
            else configured_codebooks
        )
        profile = canonical_training_profile(
            database_size=database_size,
            decoder=resolved_decoder,
            schedule=resolved_schedule,
            selected_codebooks=profile_codebooks,
        )

    tokenizer = cfg.setdefault("tokenizer", {})
    checkpoint_query = (
        _checkpoint_query_profile(checkpoint_payload)
        if checkpoint_payload is not None
        else None
    )
    if checkpoint_query is not None:
        checkpoint_codebooks = int(checkpoint_query["selected_codebooks"])
        if (
            selected_codebooks is not None
            and selected_codebooks != checkpoint_codebooks
        ):
            raise ValueError(
                "Explicit codebook selection does not match resume checkpoint"
            )
        query_profile = resolve_query_profile(checkpoint_codebooks)
        if checkpoint_query != query_profile:
            raise ValueError(
                "Resume checkpoint has an incompatible codebook/loss profile"
            )
    else:
        configured_codebooks = int(tokenizer.get("selected_codebooks", 2))
        query_profile = resolve_query_profile(
            selected_codebooks
            if selected_codebooks is not None
            else configured_codebooks
        )
    tokenizer["selected_codebooks"] = query_profile["selected_codebooks"]

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
    train["id_digit_weight"] = query_profile["id_digit_weight"]
    train["max_steps"] = profile["schedule"]["max_steps"]
    train["schedule"] = {
        key: value for key, value in profile["schedule"].items() if key != "max_steps"
    }
    train["auxiliary"] = deepcopy(profile["auxiliary"])
    lr_profile = profile.get("learning_rate_schedule")
    if lr_profile is not None:
        train["warmup_steps"] = int(lr_profile["warmup_steps"])
        train["learning_rate_schedule"] = deepcopy(lr_profile)
    elif checkpoint_payload is not None:
        historical_train = checkpoint_payload.get("hyper_parameters", {}).get(
            "train", {}
        )
        train["warmup_steps"] = int(
            historical_train.get("warmup_steps", train.get("warmup_steps", 200))
        )
        train["learning_rate_schedule"] = {"policy": "legacy_warmup_cosine_v1"}
    else:
        train["learning_rate_schedule"] = {"policy": "legacy_warmup_cosine_v1"}
    cfg["resolved_training_profile"] = profile
    cfg["resolved_query_profile"] = query_profile
    return cfg
