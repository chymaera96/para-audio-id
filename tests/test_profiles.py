from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from para_audio_id.audio_lm.losses import distillation_weight
from para_audio_id.audio_lm.noise import resolved_augmentation_schedule
from para_audio_id.audio_lm.profiles import (
    LOSS_PROTOCOL,
    NEW_TRAINING_PROTOCOL,
    SCALE_VARIANT,
    canonical_training_profile,
    cohort_manifest,
    historical_checkpoint_profile,
    resolve_training_config,
    schedule_profile,
)
from para_audio_id.audio_lm.training import learning_rate_multiplier


def base_config() -> dict:
    return {
        "tokenizer": {"selected_codebooks": 8},
        "data": {
            "database_size": 100_000,
            "room_ir": {
                "training_root": "train",
                "validation_root": "validation",
                "past_context_duration": 2.0,
            },
        },
        "model": {},
        "train": {"distillation": {"maximum_weight": 0.1}},
        "trainer": {"devices": 4},
    }


def test_scale_profile_fixes_probe_selected_parallelism_and_exposure_budget():
    profile = canonical_training_profile(
        database_size=100_000,
        decoder="medium",
        schedule="noise-rir",
        selected_codebooks=8,
        devices=4,
    )
    assert cohort_manifest(100_000) == "data/training_tracks_100k.json"
    assert profile["version"] == 11
    assert profile["variant"] == SCALE_VARIANT
    assert profile["decoder"] == {
        "name": "medium",
        "num_layers": 24,
        "hidden_size": 1024,
        "num_attention_heads": 16,
    }
    assert profile["parallelism"] == {
        "protocol": "scale_probe_fixed_16_per_gpu_4rank_v1",
        "world_size": 4,
        "tracks_per_device_microbatch": 16,
        "documents_per_device_microbatch": 32,
        "accumulate_grad_batches": 1,
        "global_tracks_per_optimizer_step": 64,
        "global_documents_per_optimizer_step": 128,
    }
    assert profile["exposure_budget"] == {
        "target_track_selections": 72_000_000,
        "reference_global_tracks_per_optimizer_step": 80,
        "resolved_global_tracks_per_optimizer_step": 64,
        "scaling": "ceil(reference_step * 80 / 64)",
    }
    assert (
        profile["schedule"]["max_steps"]
        * profile["parallelism"]["global_tracks_per_optimizer_step"]
        == 72_000_000
    )
    assert profile["operational_intervals"] == {
        "monitor_steps": 5_000,
        "checkpoint_steps": 10_000,
    }
    assert profile["schedule"] == {
        "name": "noise-rir",
        "protocol": NEW_TRAINING_PROTOCOL,
        "loss_protocol": LOSS_PROTOCOL,
        "max_steps": 1_125_000,
        "snr_bin_probabilities": [0.40, 0.30, 0.20, 0.10],
        "exact_zero_fraction_in_first_bin": 0.25,
        "curriculum": "tc12_noise_rir_curriculum_v1",
        "clean_until_step": 50_000,
        "degradation_ramp_until_step": 150_000,
        "combined_ramp_until_step": 300_000,
    }
    assert profile["distillation"]["weight_schedule"] == {
        "zero_until_step": 75_000,
        "ramp_until_step": 150_000,
    }
    assert profile["learning_rate_schedule"] == {
        "policy": "tc18_warmup_hold_linear_cosine_v1",
        "warmup_steps": 500,
        "hold_until_step": 300_000,
        "linear_decay_until_step": 700_000,
        "final_learning_rate_ratio": 0.05,
    }


def test_scale_config_defaults_to_fixed_profile():
    resolved = resolve_training_config(base_config())
    assert resolved["resolved_training_profile"]["variant"] == SCALE_VARIANT
    assert resolved["data"]["training_tracks_manifest"] == ("data/training_tracks_100k.json")
    assert resolved["model"]["num_layers"] == 24
    assert resolved["model"]["hidden_size"] == 1024
    assert resolved["model"]["num_attention_heads"] == 16
    assert resolved["train"]["tracks_per_microbatch"] == 16
    assert resolved["trainer"]["devices"] == 4
    assert resolved["trainer"]["accumulate_grad_batches"] == 1
    assert resolved["train"]["max_steps"] == 1_125_000
    assert resolved["train"]["evaluation_interval"] == 5_000
    assert resolved["train"]["checkpoint_interval"] == 10_000


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"database_size": 25_000}, "database_size=100000"),
        ({"decoder": "small"}, "medium decoder"),
        ({"schedule": "noise"}, "noise-rir schedule"),
        ({"selected_codebooks": 6}, "all eight MuQ codebooks"),
        ({"distillation_weight": -0.1}, "non-negative"),
        ({"devices": 2}, "exactly 4 devices"),
    ],
)
def test_scale_rejects_other_profiles(kwargs, message):
    values = {
        "database_size": 100_000,
        "decoder": "medium",
        "schedule": "noise-rir",
        "selected_codebooks": 8,
        "devices": 4,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        canonical_training_profile(**values)


def test_scale_resume_inherits_profile_and_rejects_overrides(tmp_path):
    profile = canonical_training_profile(
        database_size=100_000,
        decoder="medium",
        schedule="noise-rir",
        distillation_weight=0.0,
        devices=4,
    )
    path = tmp_path / "scale.ckpt"
    torch.save(
        {
            "resolved_training_profile": profile,
            "tokenizer_spec": {"selected_codebooks": 8},
            "query_spec": {"id_digit_weight": 32.0},
        },
        path,
    )
    resolved = resolve_training_config(base_config(), checkpoint=path)
    assert resolved["resolved_training_profile"] == profile
    assert resolved["train"]["distillation"]["maximum_weight"] == 0.0
    assert resolved["train"]["tracks_per_microbatch"] == 16
    assert resolved["trainer"]["accumulate_grad_batches"] == 1
    with pytest.raises(ValueError, match="distillation weight"):
        resolve_training_config(base_config(), checkpoint=path, distillation_weight=0.1)
    with pytest.raises(ValueError, match="database size"):
        resolve_training_config(base_config(), checkpoint=path, database_size=25_000)
    with pytest.raises(ValueError, match="device count"):
        resolve_training_config(base_config(), checkpoint=path, devices=1)
    with pytest.raises(ValueError, match="training profile"):
        resolve_training_config(base_config(), checkpoint=path, decoder="small")


def test_historical_tc18_remains_evaluable_but_cannot_resume(tmp_path):
    old = deepcopy(
        canonical_training_profile(
            database_size=100_000,
            decoder="medium",
            schedule="noise-rir",
            devices=4,
        )
    )
    old["version"] = 10
    old["variant"] = "tc18-two-second-eight-codebook-logit-distillation"
    payload = {"resolved_training_profile": old}
    assert historical_checkpoint_profile(payload) == old
    path = tmp_path / "tc18.ckpt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="evaluation-only"):
        resolve_training_config(base_config(), checkpoint=path)


def test_scale_noise_rir_boundaries_preserve_identity_exposure():
    profile = schedule_profile("noise-rir", 100_000)
    expected = {
        49_999: (1.0, 0.0, 0.0, 0.0),
        50_000: (1.0, 0.0, 0.0, 0.0),
        100_000: (0.70, 0.15, 0.15, 0.0),
        150_000: (0.40, 0.30, 0.30, 0.0),
        225_000: (0.25, 0.325, 0.30, 0.125),
        300_000: (0.10, 0.35, 0.30, 0.25),
        1_125_000: (0.10, 0.35, 0.30, 0.25),
    }
    for step, probabilities in expected.items():
        resolved = resolved_augmentation_schedule(step, profile)
        assert (
            resolved.clean_probability,
            resolved.noise_probability,
            resolved.rir_probability,
            resolved.noise_rir_probability,
        ) == pytest.approx(probabilities)


@pytest.mark.parametrize(
    ("step", "expected"),
    [(75_000, 0.0), (112_500, 0.05), (150_000, 0.1), (1_125_000, 0.1)],
)
def test_scale_distillation_schedule(step, expected):
    profile = canonical_training_profile(
        database_size=100_000,
        decoder="medium",
        schedule="noise-rir",
        devices=4,
    )
    boundaries = profile["distillation"]["weight_schedule"]
    assert distillation_weight(
        step,
        maximum_weight=0.1,
        zero_until_step=boundaries["zero_until_step"],
        ramp_until_step=boundaries["ramp_until_step"],
    ) == pytest.approx(expected)


def test_scale_learning_rate_schedule_preserves_exposure_boundaries():
    profile = canonical_training_profile(
        database_size=100_000,
        decoder="medium",
        schedule="noise-rir",
        devices=4,
    )
    train = {
        "max_steps": 1_125_000,
        "warmup_steps": 500,
        "learning_rate_schedule": profile["learning_rate_schedule"],
    }
    expected = {
        0: 0.0,
        250: 0.5,
        500: 1.0,
        300_000: 1.0,
        500_000: 0.75,
        700_000: 0.5,
        912_500: 0.275,
        1_125_000: 0.05,
    }
    for step, multiplier in expected.items():
        assert learning_rate_multiplier(step, train) == pytest.approx(multiplier)
