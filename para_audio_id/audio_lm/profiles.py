from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch


SUPPORTED_DATABASE_SIZES = (25_000, 100_000)
REFERENCE_DATABASE_SIZE = 25_000
DECODER_PROFILES = {
    "small": {"num_layers": 12, "hidden_size": 768, "num_attention_heads": 12},
}
SCHEDULE_NAMES = ("noise-rir",)
SUPPORTED_SELECTED_CODEBOOKS = (8,)
TC18_ID_DIGIT_WEIGHT = 32.0
DEFAULT_DISTILLATION_WEIGHT = 0.10
NEW_TRAINING_PROTOCOL = "tc18_two_second_eight_codebook_logit_distillation_v1"
LOSS_PROTOCOL = NEW_TRAINING_PROTOCOL
TC12_CURRICULUM = "tc12_noise_rir_curriculum_v1"
TC18_LR_POLICY = "tc18_warmup_hold_linear_cosine_v1"


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
    if database_size not in SUPPORTED_DATABASE_SIZES:
        raise ValueError(
            f"database_size must be one of {SUPPORTED_DATABASE_SIZES}, "
            f"got {database_size}"
        )
    return value * database_size // REFERENCE_DATABASE_SIZE


def schedule_profile(name: str, database_size: int) -> dict[str, Any]:
    if name not in SCHEDULE_NAMES:
        raise ValueError(f"schedule must be one of {SCHEDULE_NAMES}, got {name!r}")
    common = {
        "name": name,
        "protocol": NEW_TRAINING_PROTOCOL,
        "loss_protocol": LOSS_PROTOCOL,
        "max_steps": _scaled(225_000, database_size),
        "snr_bin_probabilities": [0.40, 0.30, 0.20, 0.10],
        "exact_zero_fraction_in_first_bin": 0.25,
    }
    return {
        **common,
        "curriculum": TC12_CURRICULUM,
        "clean_until_step": _scaled(10_000, database_size),
        "degradation_ramp_until_step": _scaled(30_000, database_size),
        "combined_ramp_until_step": _scaled(60_000, database_size),
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
    selected_codebooks: int = 8,
    distillation_weight: float = DEFAULT_DISTILLATION_WEIGHT,
) -> dict[str, Any]:
    if database_size not in SUPPORTED_DATABASE_SIZES:
        raise ValueError(
            f"tc18 requires database_size in {SUPPORTED_DATABASE_SIZES}"
        )
    if decoder != "small":
        raise ValueError("tc18 requires the small decoder")
    if schedule != "noise-rir":
        raise ValueError(
            "tc18 requires the noise-rir schedule"
        )
    if selected_codebooks != 8:
        raise ValueError("tc18 requires all eight MuQ codebooks")
    if not math.isfinite(distillation_weight) or distillation_weight < 0:
        raise ValueError("distillation_weight must be finite and non-negative")
    profile = {
        "version": 9,
        "variant": "tc18-two-second-eight-codebook-logit-distillation",
        "database_size": database_size,
        "training_tracks_manifest": cohort_manifest(database_size),
        "decoder": decoder_profile(decoder),
        "schedule": schedule_profile(schedule, database_size),
        "distillation": {
            "protocol": LOSS_PROTOCOL,
            "temperature": 2.0,
            "maximum_weight": float(distillation_weight),
            "weight_schedule": {
                "zero_until_step": _scaled(15_000, database_size),
                "ramp_until_step": _scaled(30_000, database_size),
            },
            "target_positions": "five_next_identifier_digits",
            "vocabulary_scope": "digit_tokens_only",
            "clean_teacher_detached": True,
        },
    }
    profile["learning_rate_schedule"] = {
        "policy": TC18_LR_POLICY,
        "warmup_steps": 500,
        "hold_until_step": _scaled(60_000, database_size),
        "linear_decay_until_step": _scaled(140_000, database_size),
        "final_learning_rate_ratio": 0.05,
    }
    return profile


def historical_checkpoint_profile(checkpoint: dict) -> dict[str, Any]:
    stored = checkpoint.get("resolved_training_profile")
    if (
        stored is None
        or stored.get("variant")
        != "tc18-two-second-eight-codebook-logit-distillation"
    ):
        raise ValueError(
            "Only tc18 two-second eight-codebook checkpoints can be "
            "resumed on this branch"
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
                "id_digit_weight", TC18_ID_DIGIT_WEIGHT
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
        "id_digit_weight": TC18_ID_DIGIT_WEIGHT,
    }


def resolve_training_config(
    config: dict[str, Any],
    *,
    decoder: str | None = None,
    schedule: str | None = None,
    selected_codebooks: int | None = None,
    distillation_weight: float | None = None,
    database_size: int | None = None,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    if distillation_weight is not None and (
        not math.isfinite(distillation_weight) or distillation_weight < 0
    ):
        raise ValueError("distillation_weight must be finite and non-negative")
    cfg = deepcopy(config)
    data = cfg.setdefault("data", {})
    configured_database_size = int(
        data.get("database_size", data.get("max_training_tracks", 0))
    )
    if database_size is not None:
        configured_database_size = int(database_size)
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
        checkpoint_database_size = int(resumed["database_size"])
        if (
            database_size is not None
            and int(database_size) != checkpoint_database_size
        ):
            raise ValueError(
                "Explicit database size does not match resume checkpoint"
            )
        resolved_database_size = checkpoint_database_size
        resolved_decoder = resumed["decoder"]["name"] if decoder is None else decoder
        resolved_schedule = resumed["schedule"]["name"] if schedule is None else schedule
    else:
        resolved_database_size = configured_database_size
        resolved_decoder = decoder or "small"
        resolved_schedule = schedule or "noise-rir"
    if resumed is not None:
        if resolved_decoder != resumed["decoder"]["name"] or (
            resolved_schedule != resumed["schedule"]["name"]
        ):
            raise ValueError("Explicit training profile does not match resume checkpoint")
        saved_distillation_weight = float(
            resumed["distillation"]["maximum_weight"]
        )
        if (
            distillation_weight is not None
            and distillation_weight != saved_distillation_weight
        ):
            raise ValueError(
                "Explicit distillation weight does not match resume checkpoint"
            )
        profile = deepcopy(resumed)
    else:
        configured_codebooks = int(
            cfg.setdefault("tokenizer", {}).get("selected_codebooks", 8)
        )
        profile_codebooks = (
            selected_codebooks
            if selected_codebooks is not None
            else configured_codebooks
        )
        profile = canonical_training_profile(
            database_size=resolved_database_size,
            decoder=resolved_decoder,
            schedule=resolved_schedule,
            selected_codebooks=profile_codebooks,
            distillation_weight=(
                float(distillation_weight)
                if distillation_weight is not None
                else float(
                    cfg.setdefault("train", {})
                    .get("distillation", {})
                    .get("maximum_weight", DEFAULT_DISTILLATION_WEIGHT)
                )
            ),
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
        configured_codebooks = int(tokenizer.get("selected_codebooks", 8))
        query_profile = resolve_query_profile(
            selected_codebooks
            if selected_codebooks is not None
            else configured_codebooks
        )
    tokenizer["selected_codebooks"] = query_profile["selected_codebooks"]

    data["database_size"] = resolved_database_size
    data["max_training_tracks"] = resolved_database_size
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
    train["distillation"] = deepcopy(profile["distillation"])
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
