from __future__ import annotations

import pytest
import torch

from para_audio_id.audio_lm.losses import relative_cosine_margin
from para_audio_id.audio_lm.noise import resolved_augmentation_schedule
from para_audio_id.audio_lm.profiles import (
    canonical_training_profile,
    cohort_manifest,
    historical_checkpoint_profile,
    resolve_training_config,
    schedule_profile,
)
from para_audio_id.audio_lm.training import learning_rate_multiplier


@pytest.mark.parametrize(
    ("size", "manifest", "total", "clean", "ramp"),
    [
        (10_000, "data/training_tracks_10k.json", 70_000, 20_000, 25_000),
        (25_000, "data/training_tracks_25k.json", 175_000, 50_000, 62_500),
        (100_000, "data/training_tracks_100k.json", 700_000, 200_000, 250_000),
    ],
)
def test_catalogue_profiles_scale_by_exposure(size, manifest, total, clean, ramp):
    assert cohort_manifest(size) == manifest
    noise = schedule_profile("noise", size)
    assert noise["max_steps"] == total
    assert noise["clean_until_step"] == clean
    assert noise["noise_ramp_until_step"] == ramp
    rir = schedule_profile("noise-rir", size)
    assert rir["max_steps"] == total
    assert rir["clean_until_step"] == 4_000 * size // 10_000
    assert rir["degradation_ramp_until_step"] == 12_000 * size // 10_000
    assert rir["combined_ramp_until_step"] == 24_000 * size // 10_000


def test_profile_defaults_and_decoder_override():
    base = {
        "data": {
            "database_size": 10_000,
            "room_ir": {
                "training_root": "train",
                "validation_root": "validation",
                "past_context_duration": 2.0,
            },
        },
        "model": {},
        "train": {},
    }
    default = resolve_training_config(base)
    assert default["resolved_training_profile"] == canonical_training_profile(
        database_size=10_000, decoder="small", schedule="noise"
    )
    medium = resolve_training_config(base, decoder="medium", schedule="noise-rir")
    assert medium["model"]["num_layers"] == 24
    assert medium["model"]["hidden_size"] == 1024
    assert medium["model"]["num_attention_heads"] == 16


@pytest.mark.parametrize(
    ("protocol", "layers", "width", "heads", "size", "decoder", "schedule"),
    [
        (
            "token_budget_matched_two_second_noise_consistency_v1",
            12,
            768,
            12,
            10_000,
            "small",
            "noise",
        ),
        (
            "token_budget_matched_two_second_noise_consistency_v1",
            24,
            1024,
            16,
            10_000,
            "medium",
            "noise",
        ),
        (
            "online_random_crop_noise_rir_consistency_25k_v1",
            12,
            768,
            12,
            25_000,
            "small",
            "noise-rir",
        ),
    ],
)
def test_historical_checkpoint_profiles(
    protocol, layers, width, heads, size, decoder, schedule
):
    checkpoint = {
        "training_protocol": protocol,
        "training_track_ids": [str(index) for index in range(size)],
        "model_config": {
            "num_layers": layers,
            "hidden_size": width,
            "num_attention_heads": heads,
        },
    }
    profile = historical_checkpoint_profile(checkpoint)
    assert profile["decoder"]["name"] == decoder
    assert profile["schedule"]["name"] == schedule
    assert profile["database_size"] == size


def test_resume_inherits_profile_and_rejects_explicit_override(tmp_path):
    profile = canonical_training_profile(
        database_size=25_000, decoder="small", schedule="noise-rir"
    )
    path = tmp_path / "checkpoint.ckpt"
    torch.save({"resolved_training_profile": profile}, path)
    base = {
        "data": {
            "database_size": 10_000,
            "room_ir": {
                "training_root": "train",
                "validation_root": "validation",
                "past_context_duration": 2.0,
            },
        },
        "model": {},
        "train": {},
    }
    resolved = resolve_training_config(base, checkpoint=path)
    assert resolved["resolved_training_profile"] == profile
    assert resolved["data"]["database_size"] == 25_000
    with pytest.raises(ValueError, match="does not match resume checkpoint"):
        resolve_training_config(base, decoder="medium", checkpoint=path)


def test_resume_inherits_historical_two_codebook_query_profile(tmp_path):
    profile = canonical_training_profile(
        database_size=10_000, decoder="small", schedule="noise"
    )
    path = tmp_path / "two-codebook.ckpt"
    torch.save(
        {
            "resolved_training_profile": profile,
            "tokenizer_spec": {"selected_codebooks": 2},
            "query_spec": {"id_digit_weight": 8.0},
        },
        path,
    )
    base = {
        "tokenizer": {"selected_codebooks": 1},
        "data": {"database_size": 10_000},
        "model": {},
        "train": {},
    }
    resolved = resolve_training_config(base, checkpoint=path)
    assert resolved["tokenizer"]["selected_codebooks"] == 2
    assert resolved["train"]["id_digit_weight"] == 8.0
    assert resolved["resolved_query_profile"] == {
        "selected_codebooks": 2,
        "id_digit_weight": 8.0,
    }
    with pytest.raises(ValueError, match="codebook selection"):
        resolve_training_config(base, selected_codebooks=1, checkpoint=path)


def test_noise_schedule_never_emits_rir():
    profile = schedule_profile("noise", 100_000)
    for step in (0, 199_999, 200_000, 225_000, 250_000, 699_999):
        resolved = resolved_augmentation_schedule(step, profile)
        assert resolved.rir_probability == 0
        assert resolved.noise_rir_probability == 0


def test_25k_noise_rir_profile_exactly_matches_tc12_boundaries():
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


def test_tc12_rir_severity_and_learning_rate_schedule():
    profile = canonical_training_profile(
        database_size=25_000,
        decoder="small",
        schedule="noise-rir",
        selected_codebooks=2,
    )
    assert profile["variant"] == "tc12-cb2"
    assert profile["schedule"]["max_steps"] == 225_000
    schedule = profile["schedule"]
    assert resolved_augmentation_schedule(
        10_000, schedule
    ).rir_severity_quantile == pytest.approx(1 / 3)
    assert resolved_augmentation_schedule(
        30_000, schedule
    ).rir_severity_quantile == pytest.approx(2 / 3)
    assert resolved_augmentation_schedule(
        60_000, schedule
    ).rir_severity_quantile == pytest.approx(1.0)
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


def test_tc12_one_codebook_keeps_original_175k_length():
    profile = canonical_training_profile(
        database_size=25_000,
        decoder="small",
        schedule="noise-rir",
        selected_codebooks=1,
    )
    assert "variant" not in profile
    assert profile["schedule"]["max_steps"] == 175_000


def test_relative_cosine_margin_and_denominator_stabilization():
    same = torch.tensor(0.8)
    different = torch.tensor(0.5)
    assert float(relative_cosine_margin(same, different)) == pytest.approx(0.6)
    stabilized = relative_cosine_margin(torch.tensor(1.0), torch.tensor(1.0))
    assert torch.isfinite(stabilized)
