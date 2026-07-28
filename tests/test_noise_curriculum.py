import numpy as np
import pytest
import soundfile as sf
import torch

from para_audio_id.audio_lm.noise import (
    BackgroundNoiseAssets,
    background_noise_schedule,
    deterministic_noise_parameters,
    mix_background_noise,
    stable_uniform,
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
