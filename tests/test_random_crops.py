import numpy as np
import soundfile as sf

from para_audio_id.catalogue import CatalogueRecord
from para_audio_id.audio_lm.noise import BackgroundNoiseAssets
from para_audio_id.audio_lm.random_crops import (
    OnlineTrackBatchSampler,
    OnlineTrackDataset,
    RandomCropCollator,
    RandomEvaluationCollator,
    make_tc6_evaluation_manifest,
    random_start_sample,
)


def records(count=4, duration=10.0):
    return [
        CatalogueRecord(
            path=f"{index}.wav",
            track_id=f"track-{index}",
            code=f"{index:05d}",
            duration=duration,
        )
        for index in range(count)
    ]


def test_random_starts_are_deterministic_in_range_and_role_specific():
    record = records(1)[0]
    kwargs = {
        "sample_rate": 24_000,
        "crop_samples": 48_000,
        "seed": 7,
        "optimizer_step": 123,
        "batch_idx": 4,
        "pair_slot": 2,
    }
    first = random_start_sample(record, role=0, **kwargs)
    repeated = random_start_sample(record, role=0, **kwargs)
    second = random_start_sample(record, role=1, **kwargs)
    assert first == repeated
    assert 0 <= first <= 192_000
    assert first != second


def test_tc6_monitor_manifest_is_stable_balanced_and_reserved():
    selected = records(4)
    first = make_tc6_evaluation_manifest(
        selected, sample_rate=24_000, crop_duration=2.0, seed=9
    )
    repeated = make_tc6_evaluation_manifest(
        selected, sample_rate=24_000, crop_duration=2.0, seed=9
    )
    assert first == repeated
    assert len(first) == 3 * len(selected)
    assert all(row["crop_duration"] == 2.0 for row in first)
    assert {row["view_type"] for row in first} == {
        "canonical",
        "shifted",
        "heldout",
    }
    record = selected[0]
    reserved = {
        row["start_sample"] for row in first if row["track_id"] == record.track_id
    }
    training = random_start_sample(
        record,
        sample_rate=24_000,
        crop_samples=48_000,
        seed=9,
        optimizer_step=0,
        batch_idx=0,
        pair_slot=0,
        role=0,
        reserved_sample=reserved,
    )
    assert training not in reserved


def test_online_track_sampler_replays_exact_batches():
    dataset = OnlineTrackDataset(records(10))
    first = OnlineTrackBatchSampler(
        dataset,
        tracks_per_microbatch=4,
        accumulation_steps=2,
        seed=5,
        catalogue_pass=3,
    )
    resumed = OnlineTrackBatchSampler(
        dataset,
        tracks_per_microbatch=4,
        accumulation_steps=2,
        seed=5,
        catalogue_pass=3,
    )
    assert list(first) == list(resumed)
    assert len(first) % 2 == 0


def test_clean_collator_produces_two_distinct_crops_per_identity(tmp_path):
    audio_root = tmp_path / "audio"
    train_noise = tmp_path / "noise_train"
    validation_noise = tmp_path / "noise_validation"
    audio_root.mkdir()
    train_noise.mkdir()
    validation_noise.mkdir()
    waveform = np.sin(
        2 * np.pi * 220 * np.arange(10 * 8_000, dtype=np.float32) / 8_000
    )
    selected = records(4)
    for index in range(4):
        sf.write(audio_root / f"{index}.wav", waveform, 8_000)
    sf.write(train_noise / "train.wav", waveform[:40_000], 8_000)
    sf.write(validation_noise / "validation.wav", waveform[40_000:], 8_000)
    assets = BackgroundNoiseAssets(
        train_noise, validation_noise, sample_rate=8_000, samples=16_000
    )
    dataset = OnlineTrackDataset(selected)
    sampler = OnlineTrackBatchSampler(
        dataset,
        tracks_per_microbatch=4,
        accumulation_steps=1,
        seed=3,
    )
    examples = [dataset[index] for index in next(iter(sampler))]
    batch = RandomCropCollator(
        records=selected,
        audio_root=audio_root,
        noise_assets=assets,
        sample_rate=8_000,
        crop_duration=2.0,
        seed=3,
        reserved_starts={},
    )(examples)
    assert batch["waveforms"].shape == (8, 16_000)
    for offset in range(0, 8, 2):
        assert (
            batch["metadata"][offset]["track_id"]
            == batch["metadata"][offset + 1]["track_id"]
        )
        assert (
            batch["metadata"][offset]["start_sample"]
            != batch["metadata"][offset + 1]["start_sample"]
        )
        assert not batch["metadata"][offset + 1]["is_noisy"]


def test_noisy_pairs_share_start_and_bad_identity_is_replaced(tmp_path):
    audio_root = tmp_path / "audio"
    train_noise = tmp_path / "noise_train"
    validation_noise = tmp_path / "noise_validation"
    audio_root.mkdir()
    train_noise.mkdir()
    validation_noise.mkdir()
    waveform = np.sin(
        2 * np.pi * 220 * np.arange(10 * 8_000, dtype=np.float32) / 8_000
    )
    selected = records(5)
    for index in range(1, 5):
        sf.write(audio_root / f"{index}.wav", waveform, 8_000)
    sf.write(train_noise / "train.wav", waveform[:40_000], 8_000)
    sf.write(validation_noise / "validation.wav", waveform[40_000:], 8_000)
    assets = BackgroundNoiseAssets(
        train_noise, validation_noise, sample_rate=8_000, samples=16_000
    )
    dataset = OnlineTrackDataset(selected)
    examples = [
        {
            **dataset[index],
            "optimizer_step": 25_000,
            "batch_idx": 0,
            "pair_slot": slot,
        }
        for slot, index in enumerate(range(4))
    ]
    batch = RandomCropCollator(
        records=selected,
        audio_root=audio_root,
        noise_assets=assets,
        sample_rate=8_000,
        crop_duration=2.0,
        seed=3,
        reserved_starts={},
    )(examples)
    assert batch["replacements"]
    assert batch["failed_crop_attempts"]
    assert batch["skipped_documents"] == 0
    assert len(batch["snr_bins"]) == sum(
        row["is_noisy"] for row in batch["metadata"]
    )
    assert len({row["track_id"] for row in batch["metadata"]}) == 4
    for offset in range(0, 8, 2):
        first, second = batch["metadata"][offset : offset + 2]
        assert first["track_id"] == second["track_id"]
        if second["is_noisy"]:
            assert first["start_sample"] == second["start_sample"]


def test_short_track_starts_at_zero_and_is_padded(tmp_path):
    record = records(1, duration=1.0)[0]
    assert (
        random_start_sample(
            record,
            sample_rate=8_000,
            crop_samples=16_000,
            seed=1,
            optimizer_step=2,
            batch_idx=3,
            pair_slot=0,
            role=0,
        )
        == 0
    )


def test_monitor_collator_skips_invalid_crop_without_aborting(tmp_path):
    audio_root = tmp_path / "audio"
    train_noise = tmp_path / "noise_train"
    validation_noise = tmp_path / "noise_validation"
    audio_root.mkdir()
    train_noise.mkdir()
    validation_noise.mkdir()
    waveform = np.sin(
        2 * np.pi * 220 * np.arange(6 * 8_000, dtype=np.float32) / 8_000
    )
    sf.write(audio_root / "valid.wav", waveform, 8_000)
    sf.write(train_noise / "train.wav", waveform[:40_000], 8_000)
    sf.write(validation_noise / "validation.wav", waveform[:40_000], 8_000)
    assets = BackgroundNoiseAssets(
        train_noise, validation_noise, sample_rate=8_000, samples=16_000
    )
    common = {
        "source_duration": 6.0,
        "start_sample": 0,
        "start": 0.0,
        "crop_duration": 2.0,
        "view_type": "canonical",
    }
    batch = RandomEvaluationCollator(
        audio_root=audio_root,
        noise_assets=assets,
        sample_rate=8_000,
        seed=3,
    )(
        [
            {
                **common,
                "track_id": "valid",
                "code": "00001",
                "source_path": "valid.wav",
            },
            {
                **common,
                "track_id": "bad",
                "code": "00002",
                "source_path": "missing.wav",
            },
        ]
    )
    assert batch["clean_waveforms"].shape == (1, 16_000)
    assert batch["track_id"] == ["valid"]
    assert len(batch["skipped"]) == 1
    assert batch["skipped"][0]["track_id"] == "bad"
