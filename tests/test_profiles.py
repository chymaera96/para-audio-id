from __future__ import annotations

import pytest
import torch

from para_audio_id.audio_lm.losses import distillation_weight
from para_audio_id.audio_lm.noise import resolved_augmentation_schedule
from para_audio_id.audio_lm.profiles import (
    NEW_TRAINING_PROTOCOL,
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
            "database_size": 25_000,
            "room_ir": {
                "training_root": "train",
                "validation_root": "validation",
                "past_context_duration": 2.0,
            },
        },
        "model": {},
        "train": {"distillation": {"maximum_weight": 0.1}},
    }


def test_tc18_25k_profile_resolves_defaults_unchanged():
    profile = canonical_training_profile(
        database_size=25_000,
        decoder="small",
        schedule="noise-rir",
        selected_codebooks=8,
    )
    assert cohort_manifest(25_000) == "data/training_tracks_25k.json"
    assert profile["version"] == 10
    assert profile["variant"] == "tc18-two-second-eight-codebook-logit-distillation"
    assert profile["decoder"] == {
        "name": "small",
        "num_layers": 12,
        "hidden_size": 768,
        "num_attention_heads": 12,
    }
    assert profile["parallelism"] == {
        "protocol": "tc18_ddp_global_80_tracks_v1",
        "world_size": 1,
        "tracks_per_device_microbatch": 40,
        "accumulate_grad_batches": 2,
        "global_tracks_per_optimizer_step": 80,
    }
    assert profile["schedule"]["protocol"] == NEW_TRAINING_PROTOCOL
    assert profile["schedule"]["max_steps"] == 225_000
    assert profile["distillation"] == {
        "protocol": "tc18_two_second_eight_codebook_logit_distillation_v1",
        "temperature": 2.0,
        "maximum_weight": 0.1,
        "weight_schedule": {
            "zero_until_step": 15_000,
            "ramp_until_step": 30_000,
        },
        "target_positions": "five_next_identifier_digits",
        "vocabulary_scope": "digit_tokens_only",
        "clean_teacher_detached": True,
    }
    resolved = resolve_training_config(base_config())
    assert resolved["resolved_training_profile"] == profile
    assert resolved["resolved_query_profile"] == {
        "selected_codebooks": 8,
        "id_digit_weight": 32.0,
    }
    assert resolved["train"]["max_steps"] == 225_000
    assert resolved["train"]["warmup_steps"] == 500
    assert resolved["train"]["learning_rate_schedule"] == {
        "policy": "tc18_warmup_hold_linear_cosine_v1",
        "warmup_steps": 500,
        "hold_until_step": 60_000,
        "linear_decay_until_step": 140_000,
        "final_learning_rate_ratio": 0.05,
    }


def test_tc18_100k_profile_scales_exposure_dependent_boundaries():
    profile = canonical_training_profile(
        database_size=100_000,
        decoder="small",
        schedule="noise-rir",
        selected_codebooks=8,
    )
    assert cohort_manifest(100_000) == "data/training_tracks_100k.json"
    assert profile["schedule"] == {
        "name": "noise-rir",
        "protocol": "tc18_two_second_eight_codebook_logit_distillation_v1",
        "loss_protocol": "tc18_two_second_eight_codebook_logit_distillation_v1",
        "max_steps": 900_000,
        "snr_bin_probabilities": [0.40, 0.30, 0.20, 0.10],
        "exact_zero_fraction_in_first_bin": 0.25,
        "curriculum": "tc12_noise_rir_curriculum_v1",
        "clean_until_step": 40_000,
        "degradation_ramp_until_step": 120_000,
        "combined_ramp_until_step": 240_000,
    }
    assert profile["distillation"]["weight_schedule"] == {
        "zero_until_step": 60_000,
        "ramp_until_step": 120_000,
    }
    assert profile["learning_rate_schedule"] == {
        "policy": "tc18_warmup_hold_linear_cosine_v1",
        "warmup_steps": 500,
        "hold_until_step": 240_000,
        "linear_decay_until_step": 560_000,
        "final_learning_rate_ratio": 0.05,
    }
    resolved = resolve_training_config(base_config(), database_size=100_000)
    assert resolved["data"]["training_tracks_manifest"] == (
        "data/training_tracks_100k.json"
    )
    assert resolved["data"]["max_training_tracks"] == 100_000
    assert resolved["train"]["max_steps"] == 900_000


def test_tc18_two_gpu_profile_preserves_global_batch():
    profile = canonical_training_profile(
        database_size=100_000,
        decoder="small",
        schedule="noise-rir",
        devices=2,
    )
    assert profile["parallelism"] == {
        "protocol": "tc18_ddp_global_80_tracks_v1",
        "world_size": 2,
        "tracks_per_device_microbatch": 40,
        "accumulate_grad_batches": 1,
        "global_tracks_per_optimizer_step": 80,
    }
    resolved = resolve_training_config(base_config(), database_size=100_000, devices=2)
    assert resolved["trainer"]["devices"] == 2
    assert resolved["trainer"]["accumulate_grad_batches"] == 1
    assert resolved["train"]["tracks_per_microbatch"] == 40
    assert resolved["train"]["max_steps"] == 900_000


def test_tc18_medium_decoder_matches_capacity_profile():
    profile = canonical_training_profile(
        database_size=100_000,
        decoder="medium",
        schedule="noise-rir",
        devices=2,
    )
    assert profile["decoder"] == {
        "name": "medium",
        "num_layers": 24,
        "hidden_size": 1024,
        "num_attention_heads": 16,
    }
    resolved = resolve_training_config(
        base_config(), database_size=100_000, decoder="medium", devices=2
    )
    assert resolved["model"]["num_layers"] == 24
    assert resolved["model"]["hidden_size"] == 1024
    assert resolved["model"]["num_attention_heads"] == 16
    assert resolved["trainer"]["accumulate_grad_batches"] == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"database_size": 10_000}, "database_size in"),
        ({"decoder": "tiny"}, "decoder must be one of"),
        ({"schedule": "noise"}, "noise-rir schedule"),
        ({"selected_codebooks": 6}, "all eight MuQ codebooks"),
        ({"distillation_weight": -0.1}, "non-negative"),
        ({"devices": 4}, "devices must be one of"),
    ],
)
def test_tc18_rejects_other_training_profiles(kwargs, message):
    values = {
        "database_size": 25_000,
        "decoder": "small",
        "schedule": "noise-rir",
        "selected_codebooks": 8,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        canonical_training_profile(**values)


def test_tc18_resume_inherits_weight_and_rejects_overrides(tmp_path):
    profile = canonical_training_profile(
        database_size=25_000,
        decoder="small",
        schedule="noise-rir",
        selected_codebooks=8,
        distillation_weight=0.0,
    )
    path = tmp_path / "tc18.ckpt"
    torch.save(
        {
            "resolved_training_profile": profile,
            "tokenizer_spec": {"selected_codebooks": 8},
            "query_spec": {"id_digit_weight": 32.0},
        },
        path,
    )
    resolved = resolve_training_config(base_config(), checkpoint=path)
    assert resolved["train"]["distillation"]["maximum_weight"] == 0.0
    assert (
        historical_checkpoint_profile(torch.load(path, weights_only=False))
        == profile
    )
    with pytest.raises(ValueError, match="distillation weight"):
        resolve_training_config(
            base_config(), distillation_weight=0.1, checkpoint=path
        )
    with pytest.raises(ValueError, match="database size"):
        resolve_training_config(
            base_config(), database_size=100_000, checkpoint=path
        )
    with pytest.raises(ValueError, match="device count"):
        resolve_training_config(base_config(), devices=2, checkpoint=path)
    with pytest.raises(ValueError, match="training profile"):
        resolve_training_config(base_config(), decoder="medium", checkpoint=path)

    for version, variant in (
        (5, "tc14-logit-distillation"),
        (6, "tc15-four-codebook-logit-distillation"),
        (7, "tc16-two-second-four-codebook-logit-distillation"),
        (8, "ablate-two-second-six-codebook-logit-distillation"),
    ):
        old = tmp_path / f"old-{version}.ckpt"
        torch.save(
            {
                "resolved_training_profile": {
                    "version": version,
                    "variant": variant,
                }
            },
            old,
        )
        with pytest.raises(ValueError, match="Only tc18"):
            resolve_training_config(base_config(), checkpoint=old)


def test_legacy_single_gpu_tc18_checkpoint_remains_resumable(tmp_path):
    profile = canonical_training_profile(
        database_size=25_000,
        decoder="small",
        schedule="noise-rir",
    )
    profile["version"] = 9
    profile.pop("parallelism")
    path = tmp_path / "legacy-tc18.ckpt"
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
    assert resolved["trainer"]["devices"] == 1
    assert resolved["trainer"]["accumulate_grad_batches"] == 2


def test_tc18_noise_rir_boundaries_remain_unchanged():
    profile = schedule_profile("noise-rir", 25_000)
    expected = {
        9_999: (1.0, 0.0, 0.0, 0.0),
        10_000: (1.0, 0.0, 0.0, 0.0),
        20_000: (0.70, 0.15, 0.15, 0.0),
        30_000: (0.40, 0.30, 0.30, 0.0),
        45_000: (0.25, 0.325, 0.30, 0.125),
        60_000: (0.10, 0.35, 0.30, 0.25),
        225_000: (0.10, 0.35, 0.30, 0.25),
    }
    for step, probabilities in expected.items():
        resolved = resolved_augmentation_schedule(step, profile)
        assert (
            resolved.clean_probability,
            resolved.noise_probability,
            resolved.rir_probability,
            resolved.noise_rir_probability,
        ) == pytest.approx(probabilities)
    assert resolved_augmentation_schedule(
        10_000, profile
    ).rir_severity_quantile == pytest.approx(1 / 3)
    assert resolved_augmentation_schedule(
        30_000, profile
    ).rir_severity_quantile == pytest.approx(2 / 3)
    assert resolved_augmentation_schedule(
        60_000, profile
    ).rir_severity_quantile == pytest.approx(1.0)


def test_tc18_100k_noise_rir_boundaries_are_scaled_fourfold():
    profile = schedule_profile("noise-rir", 100_000)
    reference = schedule_profile("noise-rir", 25_000)
    for step in (0, 5_000, 10_000, 20_000, 30_000, 45_000, 60_000, 225_000):
        expected = resolved_augmentation_schedule(step, reference)
        actual = resolved_augmentation_schedule(step * 4, profile)
        assert actual == expected


@pytest.mark.parametrize(
    ("step", "expected"),
    [(15_000, 0.0), (22_500, 0.05), (30_000, 0.1), (225_000, 0.1)],
)
def test_tc18_distillation_weight_schedule(step, expected):
    assert distillation_weight(step, maximum_weight=0.1) == pytest.approx(expected)
    assert distillation_weight(step, maximum_weight=0.0) == 0.0


@pytest.mark.parametrize(
    ("step", "expected"),
    [(60_000, 0.0), (90_000, 0.05), (120_000, 0.1), (900_000, 0.1)],
)
def test_tc18_100k_distillation_weight_schedule(step, expected):
    profile = canonical_training_profile(
        database_size=100_000,
        decoder="small",
        schedule="noise-rir",
    )
    boundaries = profile["distillation"]["weight_schedule"]
    assert distillation_weight(
        step,
        maximum_weight=0.1,
        zero_until_step=boundaries["zero_until_step"],
        ramp_until_step=boundaries["ramp_until_step"],
    ) == pytest.approx(expected)


def test_tc18_learning_rate_schedule_extends_to_225k():
    profile = canonical_training_profile(
        database_size=25_000,
        decoder="small",
        schedule="noise-rir",
        selected_codebooks=8,
    )
    train = {
        "max_steps": 225_000,
        "warmup_steps": 500,
        "learning_rate_schedule": profile["learning_rate_schedule"],
    }
    expected = {
        0: 0.0,
        250: 0.5,
        500: 1.0,
        60_000: 1.0,
        100_000: 0.75,
        140_000: 0.5,
        182_500: 0.275,
        225_000: 0.05,
    }
    for step, multiplier in expected.items():
        assert learning_rate_multiplier(step, train) == pytest.approx(multiplier)


def test_tc18_100k_learning_rate_schedule_extends_to_900k():
    profile = canonical_training_profile(
        database_size=100_000,
        decoder="small",
        schedule="noise-rir",
    )
    train = {
        "max_steps": 900_000,
        "warmup_steps": 500,
        "learning_rate_schedule": profile["learning_rate_schedule"],
    }
    expected = {
        0: 0.0,
        250: 0.5,
        500: 1.0,
        240_000: 1.0,
        400_000: 0.75,
        560_000: 0.5,
        730_000: 0.275,
        900_000: 0.05,
    }
    for step, multiplier in expected.items():
        assert learning_rate_multiplier(step, train) == pytest.approx(multiplier)
