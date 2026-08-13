from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from para_audio_id.audio_lm.losses import relative_cosine_margin
from para_audio_id.audio_lm.noise import resolved_augmentation_schedule
from para_audio_id.audio_lm.profiles import (
    CAPACITY_TRAINING_PROTOCOL,
    canonical_capacity_profile,
    canonical_training_profile,
    cohort_manifest,
    historical_checkpoint_profile,
    resolve_training_config,
    resolve_capacity_config,
    schedule_profile,
)


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
    assert rir["noise_steady_until_step"] == 35_000 * size // 10_000
    assert rir["rir_ramp_until_step"] == 40_000 * size // 10_000
    assert rir["combined_ramp_until_step"] == 45_000 * size // 10_000


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


def test_noise_schedule_never_emits_rir():
    profile = schedule_profile("noise", 100_000)
    for step in (0, 199_999, 200_000, 225_000, 250_000, 699_999):
        resolved = resolved_augmentation_schedule(step, profile)
        assert resolved.rir_probability == 0
        assert resolved.noise_rir_probability == 0


def test_25k_noise_rir_profile_exactly_matches_tc11_boundaries():
    profile = schedule_profile("noise-rir", 25_000)
    expected = {
        49_999: (1.0, 0.0, 0.0, 0.0),
        62_500: (0.25, 0.75, 0.0, 0.0),
        87_500: (0.25, 0.75, 0.0, 0.0),
        100_000: (0.25, 0.55, 0.20, 0.0),
        112_500: (0.25, 0.35, 0.20, 0.20),
        175_000: (0.25, 0.35, 0.20, 0.20),
    }
    for step, probabilities in expected.items():
        resolved = resolved_augmentation_schedule(step, profile)
        assert (
            resolved.clean_probability,
            resolved.noise_probability,
            resolved.rir_probability,
            resolved.noise_rir_probability,
        ) == pytest.approx(probabilities)


def test_relative_cosine_margin_and_denominator_stabilization():
    same = torch.tensor(0.8)
    different = torch.tensor(0.5)
    assert float(relative_cosine_margin(same, different)) == pytest.approx(0.6)
    stabilized = relative_cosine_margin(torch.tensor(1.0), torch.tensor(1.0))
    assert torch.isfinite(stabilized)


@pytest.mark.parametrize(
    ("size", "steps"),
    [(10_000, 70_000), (25_000, 175_000), (50_000, 350_000), (100_000, 700_000)],
)
def test_capacity_profiles_resolve_560_exposures(size, steps):
    profile = canonical_capacity_profile(
        database_size=size,
        decoder="small",
        target_exposures=560,
        tracks_per_optimizer_step=80,
    )
    assert profile["schedule"]["protocol"] == CAPACITY_TRAINING_PROTOCOL
    assert profile["schedule"]["max_steps"] == steps
    assert profile["training_tracks_manifest"] == (
        f"data/training_tracks_{size // 1000}k.json"
    )


def test_capacity_decoder_profiles_and_defaults():
    base = {
        "data": {"database_size": 10_000},
        "model": {},
        "train": {"target_exposures": 560, "tracks_per_microbatch": 10},
        "trainer": {"accumulate_grad_batches": 8},
    }
    default = resolve_capacity_config(base)
    assert default["resolved_training_profile"]["decoder"]["name"] == "small"
    tiny = resolve_capacity_config(base, decoder="tiny")
    assert tiny["model"]["num_layers"] == 6
    assert tiny["model"]["hidden_size"] == 512
    assert tiny["model"]["num_attention_heads"] == 8
    medium = resolve_capacity_config(base, decoder="medium")
    assert medium["model"]["num_layers"] == 24


def test_capacity_resume_rejects_corruption_checkpoint(tmp_path):
    path = tmp_path / "checkpoint.ckpt"
    torch.save(
        {
            "resolved_training_profile": canonical_training_profile(
                database_size=10_000, decoder="small", schedule="noise"
            )
        },
        path,
    )
    base = {
        "data": {"database_size": 10_000},
        "model": {},
        "train": {"target_exposures": 560, "tracks_per_microbatch": 10},
        "trainer": {"accumulate_grad_batches": 8},
    }
    with pytest.raises(ValueError, match="cannot resume"):
        resolve_capacity_config(base, checkpoint=path)


def test_capacity_resume_inherits_decoder_and_rejects_mismatch(tmp_path):
    profile = canonical_capacity_profile(
        database_size=10_000,
        decoder="tiny",
        target_exposures=560,
        tracks_per_optimizer_step=80,
    )
    path = tmp_path / "capacity.ckpt"
    torch.save({"resolved_training_profile": profile}, path)
    base = {
        "data": {"database_size": 10_000},
        "model": {},
        "train": {"target_exposures": 560, "tracks_per_microbatch": 10},
        "trainer": {"accumulate_grad_batches": 8},
    }
    resumed = resolve_capacity_config(base, checkpoint=path)
    assert resumed["resolved_training_profile"] == profile
    assert resumed["model"]["num_layers"] == 6
    with pytest.raises(ValueError, match="does not match resume checkpoint"):
        resolve_capacity_config(base, decoder="small", checkpoint=path)
    wrong_database = deepcopy(base)
    wrong_database["data"]["database_size"] = 25_000
    with pytest.raises(ValueError, match="database size"):
        resolve_capacity_config(wrong_database, checkpoint=path)
