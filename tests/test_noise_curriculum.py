import numpy as np
import pytest
import soundfile as sf
import torch

from para_audio_id.audio_lm.noise import (
    BackgroundNoiseAssets,
    NoiseConsistencySchedule,
    background_noise_schedule,
    deterministic_consistency_noise_parameters,
    deterministic_augmentation_parameters,
    deterministic_noise_parameters,
    mix_background_noise,
    noise_consistency_schedule,
    stable_uniform,
    tc9_noise_consistency_schedule,
    tc11_augmentation_schedule,
)


@pytest.mark.parametrize(
    ("step", "probability", "snr"),
    [
        (0, 0.0, (None, None)),
        (19_999, 0.0, (None, None)),
        (20_000, 0.0, (20.0, 30.0)),
        (22_500, 0.125, (20.0, 30.0)),
        (25_000, 0.25, (10.0, 30.0)),
        (30_000, 0.375, (10.0, 30.0)),
        (35_000, 0.50, (0.0, 30.0)),
        (40_000, 0.625, (0.0, 30.0)),
        (45_000, 0.75, (0.0, 30.0)),
        (60_000, 0.75, (0.0, 30.0)),
    ],
)
def test_background_noise_schedule(step, probability, snr):
    schedule = background_noise_schedule(step)
    assert schedule.probability == pytest.approx(probability)
    assert (schedule.snr_min_db, schedule.snr_max_db) == snr


def test_background_mixing_hits_snr_and_prevents_clipping():
    generator = torch.Generator().manual_seed(4)
    signal = torch.randn(2, 24_000, generator=generator)
    noise = torch.randn(2, 24_000, generator=generator)
    requested = torch.tensor([0.0, 20.0])
    mixed, valid = mix_background_noise(signal, noise, requested)
    assert valid.all()
    assert float(mixed.abs().max()) <= 0.99901
    # Global anti-clipping attenuation preserves the signal-to-noise ratio.
    signal_rms = signal.square().mean(dim=1).sqrt()
    noise_rms = noise.square().mean(dim=1).sqrt()
    scale = signal_rms / (noise_rms * torch.pow(10.0, requested / 20.0))
    expected = signal + noise * scale[:, None]
    attenuation = torch.minimum(
        torch.ones(2), torch.tensor(0.999) / expected.abs().amax(dim=1)
    )
    assert torch.allclose(mixed, expected * attenuation[:, None], atol=1e-5)


def test_silent_signal_falls_back_to_clean():
    signal = torch.zeros(1, 100)
    noise = torch.ones(1, 100)
    mixed, valid = mix_background_noise(signal, noise, torch.tensor([10.0]))
    assert not valid.any()
    assert torch.equal(mixed, signal)


def test_deterministic_noise_selection_handles_zero_one_and_resume():
    keys = [11, 12, 13, 14]
    disabled, _ = deterministic_noise_parameters(
        keys,
        probability=0.0,
        snr_min_db=None,
        snr_max_db=None,
        seed=7,
        step=20,
        batch_idx=3,
    )
    enabled, snrs = deterministic_noise_parameters(
        keys,
        probability=1.0,
        snr_min_db=0.0,
        snr_max_db=30.0,
        seed=7,
        step=20,
        batch_idx=3,
    )
    resumed, resumed_snrs = deterministic_noise_parameters(
        keys,
        probability=1.0,
        snr_min_db=0.0,
        snr_max_db=30.0,
        seed=7,
        step=20,
        batch_idx=3,
    )
    assert disabled == [False] * 4
    assert enabled == resumed == [True] * 4
    assert snrs == resumed_snrs
    assert all(0.0 <= snr <= 30.0 for snr in snrs)


def test_noise_assets_loop_short_files_and_are_source_disjoint(tmp_path):
    train = tmp_path / "train"
    validation = tmp_path / "validation"
    train.mkdir()
    validation.mkdir()
    sf.write(train / "short.wav", np.linspace(-1, 1, 40, dtype=np.float32), 8_000)
    sf.write(
        validation / "heldout.wav",
        np.linspace(1, -1, 80, dtype=np.float32),
        8_000,
    )
    assets = BackgroundNoiseAssets(
        train, validation, sample_rate=8_000, samples=400
    )
    first = assets.load_training("key")
    repeated = assets.load_training("key")
    assert len(first) == 400
    assert np.array_equal(first, repeated)
    assert assets.training_fingerprint != assets.validation_fingerprint
    assert stable_uniform("same") == stable_uniform("same")
    heldout, path, offset = assets.load_validation_recipe("heldout")
    heldout_again, path_again, offset_again = assets.load_validation_recipe(
        "heldout"
    )
    assert np.array_equal(heldout, heldout_again)
    assert (path, offset) == (path_again, offset_again) == ("heldout.wav", 0)


def test_noise_assets_reject_overlapping_source_names(tmp_path):
    train = tmp_path / "train"
    validation = tmp_path / "validation"
    train.mkdir()
    validation.mkdir()
    audio = np.linspace(-1, 1, 80, dtype=np.float32)
    sf.write(train / "same.wav", audio, 8_000)
    sf.write(validation / "same.wav", audio, 8_000)
    with pytest.raises(ValueError, match="overlap"):
        BackgroundNoiseAssets(
            train, validation, sample_rate=8_000, samples=40
        )


@pytest.mark.parametrize(
    ("step", "probability", "weight", "phase"),
    [
        (0, 0.0, 0.0, "clean"),
        (19_999, 0.0, 0.0, "clean"),
        (20_000, 0.0, 0.0, "easy"),
        (25_000, 0.125, 0.025, "easy"),
        (30_000, 0.25, 0.05, "mixed"),
        (35_000, 0.375, 0.075, "mixed"),
        (40_000, 0.50, 0.10, "hardening"),
        (47_500, 0.625, 0.10, "hardening"),
        (55_000, 0.75, 0.10, "consolidation"),
        (70_000, 0.75, 0.10, "consolidation"),
    ],
)
def test_tc6_schedule_boundaries(step, probability, weight, phase):
    schedule = noise_consistency_schedule(step)
    assert schedule.probability == pytest.approx(probability)
    assert schedule.consistency_weight == pytest.approx(weight)
    assert schedule.phase == phase


def test_tc6_schedule_scales_with_nominal_steps():
    halfway = noise_consistency_schedule(17_500, max_steps=35_000)
    reference = noise_consistency_schedule(35_000, max_steps=70_000)
    assert halfway == reference


def test_tc6_final_snr_sampler_matches_bins_and_exact_zero_mass():
    schedule = NoiseConsistencySchedule(
        1.0,
        0.1,
        (0.40, 0.30, 0.20, 0.10),
        "consolidation",
    )
    _, snrs, bins = deterministic_consistency_noise_parameters(
        list(range(20_000)),
        schedule=schedule,
        seed=7,
        step=60_000,
        batch_idx=3,
    )
    counts = {name: bins.count(name) / len(bins) for name in set(bins)}
    assert counts["very_hard"] == pytest.approx(0.40, abs=0.015)
    assert counts["hard"] == pytest.approx(0.30, abs=0.015)
    assert counts["medium"] == pytest.approx(0.20, abs=0.015)
    assert counts["easy"] == pytest.approx(0.10, abs=0.015)
    assert sum(snr == 0.0 for snr in snrs) / len(snrs) == pytest.approx(
        0.10, abs=0.01
    )


def test_tc6_noise_recipe_is_resume_stable_and_changes_by_step():
    schedule = NoiseConsistencySchedule(
        1.0,
        0.1,
        (0.20, 0.30, 0.30, 0.20),
        "hardening",
    )
    first = deterministic_consistency_noise_parameters(
        [11, 12, 13, 14],
        schedule=schedule,
        seed=5,
        step=41_000,
        batch_idx=7,
    )
    resumed = deterministic_consistency_noise_parameters(
        [11, 12, 13, 14],
        schedule=schedule,
        seed=5,
        step=41_000,
        batch_idx=7,
    )
    next_step = deterministic_consistency_noise_parameters(
        [11, 12, 13, 14],
        schedule=schedule,
        seed=5,
        step=41_001,
        batch_idx=7,
    )
    assert first == resumed
    assert first != next_step


def test_snr_bins_are_only_realized_for_selected_noise():
    schedule = NoiseConsistencySchedule(
        0.0,
        0.1,
        (0.40, 0.30, 0.20, 0.10),
        "steady",
    )
    selected, snrs, bins = deterministic_consistency_noise_parameters(
        [11, 12, 13, 14],
        schedule=schedule,
        seed=5,
        step=25_000,
        batch_idx=7,
    )
    assert selected == [False] * 4
    assert snrs == [0.0] * 4
    assert bins == [None] * 4


@pytest.mark.parametrize(
    ("step", "probability", "weight", "phase"),
    [
        (0, 0.0, 0.0, "clean"),
        (19_999, 0.0, 0.0, "clean"),
        (20_000, 0.0, 0.0, "ramp"),
        (22_500, 0.375, 0.05, "ramp"),
        (24_999, 0.75 * 4_999 / 5_000, 0.10 * 4_999 / 5_000, "ramp"),
        (25_000, 0.75, 0.10, "steady"),
        (70_000, 0.75, 0.10, "steady"),
    ],
)
def test_tc9_schedule_boundaries(step, probability, weight, phase):
    schedule = tc9_noise_consistency_schedule(step)
    assert schedule.probability == pytest.approx(probability)
    assert schedule.consistency_weight == pytest.approx(weight)
    assert schedule.phase == phase


@pytest.mark.parametrize(
    ("step", "clean", "noise", "rir", "noise_rir", "weight"),
    [
        (0, 1.0, 0.0, 0.0, 0.0, 0.0),
        (50_000, 1.0, 0.0, 0.0, 0.0, 0.0),
        (56_250, 0.625, 0.375, 0.0, 0.0, 0.05),
        (62_500, 0.25, 0.75, 0.0, 0.0, 0.10),
        (87_500, 0.25, 0.75, 0.0, 0.0, 0.10),
        (93_750, 0.25, 0.65, 0.10, 0.0, 0.10),
        (100_000, 0.25, 0.55, 0.20, 0.0, 0.10),
        (106_250, 0.25, 0.45, 0.20, 0.10, 0.10),
        (112_500, 0.25, 0.35, 0.20, 0.20, 0.10),
    ],
)
def test_tc11_schedule_boundaries(step, clean, noise, rir, noise_rir, weight):
    schedule = tc11_augmentation_schedule(step)
    assert schedule.clean_probability == pytest.approx(clean)
    assert schedule.noise_probability == pytest.approx(noise)
    assert schedule.rir_probability == pytest.approx(rir)
    assert schedule.noise_rir_probability == pytest.approx(noise_rir)
    assert schedule.consistency_weight == pytest.approx(weight)
    assert sum(
        (
            schedule.clean_probability,
            schedule.noise_probability,
            schedule.rir_probability,
            schedule.noise_rir_probability,
        )
    ) == pytest.approx(1.0)


def test_tc11_augmentation_recipe_is_deterministic_and_only_noise_gets_snr():
    schedule = tc11_augmentation_schedule(112_500)
    first = deterministic_augmentation_parameters(
        list(range(10_000)), schedule=schedule, seed=4, step=112_500, batch_idx=3
    )
    resumed = deterministic_augmentation_parameters(
        list(range(10_000)), schedule=schedule, seed=4, step=112_500, batch_idx=3
    )
    categories, snrs, bins = first
    assert first == resumed
    assert all(
        ("noise" in category) == (bin is not None)
        for category, bin in zip(categories, bins, strict=True)
    )
    assert all(
        ("noise" in category) or snr == 0.0
        for category, snr in zip(categories, snrs, strict=True)
    )
