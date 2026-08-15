from __future__ import annotations

import pytest
import torch

from para_audio_id.audio_lm.losses import relative_cosine_margin
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
        "tokenizer": {"selected_codebooks": 2},
        "data": {
            "database_size": 25_000,
            "room_ir": {
                "training_root": "train",
                "validation_root": "validation",
                "past_context_duration": 2.0,
            },
        },
        "model": {},
        "train": {},
    }


def test_tc13_profile_is_fixed_and_resolves_defaults():
    profile = canonical_training_profile(
        database_size=25_000,
        decoder="small",
        schedule="noise-rir",
        selected_codebooks=2,
    )
    assert cohort_manifest(25_000) == "data/training_tracks_25k.json"
    assert profile["version"] == 3
    assert profile["variant"] == "tc13"
    assert profile["decoder"] == {
        "name": "small",
        "num_layers": 12,
        "hidden_size": 768,
        "num_attention_heads": 12,
    }
    assert profile["schedule"]["protocol"] == NEW_TRAINING_PROTOCOL
    assert profile["schedule"]["max_steps"] == 225_000
    resolved = resolve_training_config(base_config())
    assert resolved["resolved_training_profile"] == profile
    assert resolved["resolved_query_profile"] == {
        "selected_codebooks": 2,
        "id_digit_weight": 20.0,
    }
    assert resolved["train"]["max_steps"] == 225_000
    assert resolved["train"]["warmup_steps"] == 500


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"database_size": 10_000}, "database_size=25000"),
        ({"decoder": "medium"}, "small decoder"),
        ({"schedule": "noise"}, "noise-rir schedule"),
        ({"selected_codebooks": 1}, "two MuQ codebooks"),
    ],
)
def test_tc13_rejects_other_training_profiles(kwargs, message):
    values = {
        "database_size": 25_000,
        "decoder": "small",
        "schedule": "noise-rir",
        "selected_codebooks": 2,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        canonical_training_profile(**values)


def test_tc13_resume_accepts_only_exact_tc13_profile(tmp_path):
    profile = canonical_training_profile(
        database_size=25_000,
        decoder="small",
        schedule="noise-rir",
        selected_codebooks=2,
    )
    path = tmp_path / "tc13.ckpt"
    torch.save(
        {
            "resolved_training_profile": profile,
            "tokenizer_spec": {"selected_codebooks": 2},
            "query_spec": {"id_digit_weight": 20.0},
        },
        path,
    )
    resolved = resolve_training_config(base_config(), checkpoint=path)
    assert resolved["resolved_training_profile"] == profile
    assert historical_checkpoint_profile(torch.load(path, weights_only=False)) == profile
    with pytest.raises(ValueError, match="does not match resume checkpoint"):
        resolve_training_config(base_config(), decoder="medium", checkpoint=path)

    old = tmp_path / "tc12.ckpt"
    torch.save(
        {"resolved_training_profile": {"version": 2, "variant": "tc12-cb2"}},
        old,
    )
    with pytest.raises(ValueError, match="Only tc13 checkpoints"):
        resolve_training_config(base_config(), checkpoint=old)


def test_tc13_noise_rir_boundaries_and_severity():
    profile = schedule_profile("noise-rir", 25_000)
    expected = {
        9_999: (1.0, 0.0, 0.0, 0.0),
        10_000: (1.0, 0.0, 0.0, 0.0),
        20_000: (0.70, 0.15, 0.15, 0.0),
        30_000: (0.40, 0.30, 0.30, 0.0),
        45_000: (0.25, 0.325, 0.30, 0.125),
        60_000: (0.10, 0.35, 0.30, 0.25),
        140_000: (0.10, 0.35, 0.30, 0.25),
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


def test_tc13_learning_rate_schedule_extends_to_225k():
    profile = canonical_training_profile(
        database_size=25_000,
        decoder="small",
        schedule="noise-rir",
        selected_codebooks=2,
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


def test_relative_cosine_margin_and_denominator_stabilization():
    same = torch.tensor(0.8)
    different = torch.tensor(0.5)
    assert float(relative_cosine_margin(same, different)) == pytest.approx(0.6)
    stabilized = relative_cosine_margin(torch.tensor(1.0), torch.tensor(1.0))
    assert torch.isfinite(stabilized)
