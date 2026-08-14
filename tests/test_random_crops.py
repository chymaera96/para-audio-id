import json
import numpy as np
import pytest
import random
import soundfile as sf

from para_audio_id.catalogue import CatalogueRecord
from para_audio_id.audio_lm.noise import BackgroundNoiseAssets
from para_audio_id.audio_lm.evaluation import (
    _joint_manifest_configuration,
    _joint_metrics,
    _load_joint_rows,
    _load_or_create_joint_manifest,
)
from para_audio_id.audio_lm.rir import RoomImpulseResponseAssets, convolve_full_wet
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


def test_online_track_sampler_shards_identities_across_ranks():
    dataset = OnlineTrackDataset(records(80))
    rank_zero = OnlineTrackBatchSampler(
        dataset,
        tracks_per_microbatch=40,
        accumulation_steps=1,
        seed=5,
        world_size=2,
        rank=0,
    )
    rank_one = OnlineTrackBatchSampler(
        dataset,
        tracks_per_microbatch=40,
        accumulation_steps=1,
        seed=5,
        world_size=2,
        rank=1,
    )
    zero_batch = next(iter(rank_zero))
    one_batch = next(iter(rank_one))
    zero_indices = {row[0] for row in zero_batch}
    one_indices = {row[0] for row in one_batch}
    assert zero_indices == set(range(0, 80, 2))
    assert one_indices == set(range(1, 80, 2))
    assert zero_indices.isdisjoint(one_indices)
    assert {row[1] for row in zero_batch} == {0}
    assert {row[1] for row in one_batch} == {0}
    assert {row[3] for row in zero_batch} == set(range(40))
    assert {row[3] for row in one_batch} == set(range(40, 80))


def test_distributed_online_track_sampler_resume_alignment_matches_ranks():
    dataset = OnlineTrackDataset(records(160))
    samplers = [
        OnlineTrackBatchSampler(
            dataset,
            tracks_per_microbatch=40,
            accumulation_steps=1,
            seed=9,
            catalogue_pass=3,
            world_size=2,
            rank=rank,
        )
        for rank in range(2)
    ]
    for sampler in samplers:
        sampler.align_resume_position(batches_yielded=1, global_step=7)
    batches = [next(iter(sampler)) for sampler in samplers]
    assert [{row[1] for row in batch} for batch in batches] == [{6}, {6}]


def test_online_track_sampler_rebases_after_completed_epoch_resume():
    dataset = OnlineTrackDataset(records(10))
    sampler = OnlineTrackBatchSampler(
        dataset,
        tracks_per_microbatch=10,
        accumulation_steps=8,
        seed=5,
        catalogue_pass=312,
    )
    sampler.align_resume_position(batches_yielded=len(sampler), global_step=312)
    sampler.set_epoch(313)
    first_batch = next(iter(sampler))
    assert {row[1] for row in first_batch} == {312}


def test_online_track_sampler_accepts_runtime_epoch_correction():
    dataset = OnlineTrackDataset(records(10))
    sampler = OnlineTrackBatchSampler(
        dataset,
        tracks_per_microbatch=10,
        accumulation_steps=8,
        seed=5,
        catalogue_pass=313,
    )
    first_batch = next(iter(sampler))
    assert {row[1] for row in first_batch} == {313}
    sampler.optimizer_step_offset += 312 - 313
    corrected_batch = next(iter(sampler))
    assert {row[1] for row in corrected_batch} == {312}


def test_clean_collator_produces_two_distinct_crops_per_identity(tmp_path):
    audio_root = tmp_path / "audio"
    train_noise = tmp_path / "noise_train"
    validation_noise = tmp_path / "noise_validation"
    audio_root.mkdir()
    train_noise.mkdir()
    validation_noise.mkdir()
    rir_train = tmp_path / "rir_train" / "OpenAIR" / "train-room"
    rir_validation = tmp_path / "rir_validation" / "OpenAIR" / "test-room"
    rir_train.mkdir(parents=True)
    rir_validation.mkdir(parents=True)
    waveform = np.sin(
        2 * np.pi * 220 * np.arange(10 * 8_000, dtype=np.float32) / 8_000
    )
    selected = records(4)
    for index in range(4):
        sf.write(audio_root / f"{index}.wav", waveform, 8_000)
    sf.write(train_noise / "train.wav", waveform[:40_000], 8_000)
    sf.write(validation_noise / "validation.wav", waveform[40_000:], 8_000)
    impulse = np.zeros(800, dtype=np.float32)
    impulse[0] = 1.0
    sf.write(rir_train / "ir.wav", impulse, 8_000)
    sf.write(rir_validation / "ir.wav", impulse[::-1], 8_000)
    assets = BackgroundNoiseAssets(
        train_noise, validation_noise, sample_rate=8_000, samples=16_000
    )
    rir_assets = RoomImpulseResponseAssets(
        rir_train.parent.parent, rir_validation.parent.parent, sample_rate=8_000
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
        rir_assets=rir_assets,
        schedule={
            "name": "noise-rir",
            "clean_until_step": 50_000,
            "noise_ramp_until_step": 62_500,
            "noise_steady_until_step": 87_500,
            "rir_ramp_until_step": 100_000,
            "combined_ramp_until_step": 112_500,
            "consistency_weight": 0.1,
            "snr_bin_probabilities": [0.4, 0.3, 0.2, 0.1],
        },
        sample_rate=8_000,
        crop_duration=2.0,
        past_context_duration=2.0,
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


def test_capacity_clean_collator_requires_no_degradation_assets(tmp_path):
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    waveform = np.sin(
        2 * np.pi * 220 * np.arange(10 * 8_000, dtype=np.float32) / 8_000
    )
    selected = records(4)
    for index in range(4):
        sf.write(audio_root / f"{index}.wav", waveform, 8_000)
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
        noise_assets=None,
        rir_assets=None,
        schedule={"name": "clean"},
        sample_rate=8_000,
        crop_duration=2.0,
        past_context_duration=0.0,
        seed=3,
        reserved_starts={},
    )(examples)
    assert batch["waveforms"].shape == (8, 16_000)
    assert batch["categories"] == ["clean"] * 4
    assert not batch["snrs"]
    assert not batch["rir_paths"]
    for offset in range(0, 8, 2):
        anchor, secondary = batch["metadata"][offset : offset + 2]
        assert anchor["track_id"] == secondary["track_id"]
        assert anchor["start_sample"] != secondary["start_sample"]
        assert not anchor["is_noisy"] and not secondary["is_noisy"]


def test_noisy_pairs_share_start_and_bad_identity_is_replaced(tmp_path):
    audio_root = tmp_path / "audio"
    train_noise = tmp_path / "noise_train"
    validation_noise = tmp_path / "noise_validation"
    audio_root.mkdir()
    train_noise.mkdir()
    validation_noise.mkdir()
    rir_train = tmp_path / "rir_train" / "OpenAIR" / "train-room"
    rir_validation = tmp_path / "rir_validation" / "OpenAIR" / "test-room"
    rir_train.mkdir(parents=True)
    rir_validation.mkdir(parents=True)
    waveform = np.sin(
        2 * np.pi * 220 * np.arange(10 * 8_000, dtype=np.float32) / 8_000
    )
    selected = records(5)
    for index in range(1, 5):
        sf.write(audio_root / f"{index}.wav", waveform, 8_000)
    sf.write(train_noise / "train.wav", waveform[:40_000], 8_000)
    sf.write(validation_noise / "validation.wav", waveform[40_000:], 8_000)
    impulse = np.zeros(800, dtype=np.float32)
    impulse[0] = 1.0
    sf.write(rir_train / "ir.wav", impulse, 8_000)
    sf.write(rir_validation / "ir.wav", impulse[::-1], 8_000)
    assets = BackgroundNoiseAssets(
        train_noise, validation_noise, sample_rate=8_000, samples=16_000
    )
    rir_assets = RoomImpulseResponseAssets(
        rir_train.parent.parent, rir_validation.parent.parent, sample_rate=8_000
    )
    dataset = OnlineTrackDataset(selected)
    examples = [
        {
            **dataset[index],
            "optimizer_step": 62_500,
            "batch_idx": 0,
            "pair_slot": slot,
        }
        for slot, index in enumerate(range(4))
    ]
    batch = RandomCropCollator(
        records=selected,
        audio_root=audio_root,
        noise_assets=assets,
        rir_assets=rir_assets,
        schedule={
            "name": "noise-rir",
            "clean_until_step": 50_000,
            "noise_ramp_until_step": 62_500,
            "noise_steady_until_step": 87_500,
            "rir_ramp_until_step": 100_000,
            "combined_ramp_until_step": 112_500,
            "consistency_weight": 0.1,
            "snr_bin_probabilities": [0.4, 0.3, 0.2, 0.1],
        },
        sample_rate=8_000,
        crop_duration=2.0,
        past_context_duration=2.0,
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


def test_full_wet_convolution_uses_past_context_and_normalizes_peak():
    context = np.zeros(16, dtype=np.float32)
    context[7] = 1.0
    ir = np.array([1.0, 0.5], dtype=np.float32)
    result = convolve_full_wet(
        context, ir, past_context_samples=8, output_samples=8
    )
    assert result.shape == (8,)
    # The impulse is in past context, so only its causal tail reaches the query.
    assert result[0] == np.float32(0.5)
    assert np.max(np.abs(result)) == np.float32(0.5)
    assert np.allclose(result[1:], 0.0, atol=1e-6)


def test_joint_manifest_is_deterministic_and_backfills_bad_candidates(tmp_path):
    sample_rate = 8_000
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    selected_records = records(4, duration=12.0)
    track_ids = [record.track_id for record in selected_records]
    shuffled = track_ids.copy()
    random.Random(7).shuffle(shuffled)
    missing_track = shuffled[0]
    waveform = np.sin(
        2 * np.pi * 220 * np.arange(12 * sample_rate, dtype=np.float32)
        / sample_rate
    )
    for record in selected_records:
        if record.track_id != missing_track:
            sf.write(audio_root / record.path, waveform, sample_rate)
    catalogue = tmp_path / "catalogue.jsonl"
    catalogue.write_text(
        "\n".join(
            json.dumps(
                {
                    "path": record.path,
                    "track_id": record.track_id,
                    "code": record.code,
                    "duration": record.duration,
                }
            )
            for record in selected_records
        )
    )
    rir_train = tmp_path / "rir_train" / "OpenAIR" / "train-room"
    rir_test = tmp_path / "rir_test" / "OpenAIR" / "test-room"
    rir_train.mkdir(parents=True)
    rir_test.mkdir(parents=True)
    impulse = np.zeros(32, dtype=np.float32)
    impulse[0] = 1.0
    sf.write(rir_train / "train.wav", impulse, sample_rate)
    test_ir = impulse.copy()
    test_ir[1] = 0.5
    sf.write(rir_test / "test.wav", test_ir, sample_rate)
    rir_assets = RoomImpulseResponseAssets(
        rir_train.parents[1], rir_test.parents[1], sample_rate=sample_rate
    )
    checkpoint = {
        "training_track_ids": track_ids,
        "validation_probe": [],
        "tokenizer_fingerprint": "tokenizer",
        "training_corpus_fingerprint": "corpus",
        "global_step": 5,
    }
    configuration = _joint_manifest_configuration(
        checkpoint_fingerprint="checkpoint",
        checkpoint=checkpoint,
        rir_manifest=rir_assets.manifest(),
        cohort="training",
        expected_tracks=4,
        sample_tracks=2,
        sample_seed=7,
        recipe_seed=9,
        query_lengths=(2.0, 3.0, 5.0, 10.0),
        conditions=("clean", "rir"),
        beam_width=10,
        sample_rate=sample_rate,
        window_seconds=2.0,
        past_context_seconds=2.0,
    )
    cfg = {
        "data": {"catalogue": str(catalogue), "audio_root": str(audio_root)}
    }
    path = tmp_path / "evaluation.manifest.json"
    first = _load_or_create_joint_manifest(
        path=path,
        configuration=configuration,
        checkpoint=checkpoint,
        cfg=cfg,
        rir_assets=rir_assets,
    )
    repeated = _load_or_create_joint_manifest(
        path=path,
        configuration=configuration,
        checkpoint=checkpoint,
        cfg=cfg,
        rir_assets=rir_assets,
    )
    assert first == repeated
    assert len(first["queries"]) == 2
    assert len({row["track_id"] for row in first["queries"]}) == 2
    assert first["excluded_candidates"][0]["track_id"] == missing_track
    assert all(row["rir_path"] == "OpenAIR/test-room/test.wav" for row in first["queries"])
    changed = {**configuration, "recipe_seed": 10}
    with pytest.raises(ValueError, match="does not match"):
        _load_or_create_joint_manifest(
            path=path,
            configuration=changed,
            checkpoint=checkpoint,
            cfg=cfg,
            rir_assets=rir_assets,
        )


def test_joint_metrics_and_jsonl_resume_validation(tmp_path):
    rows = [
        {
            "protocol_fingerprint": "fingerprint",
            "status": "ok",
            "track_id": "a",
            "query_seconds": 2.0,
            "condition": "clean",
            "correct_rank": 1,
            "latency_seconds": 1.0,
        },
        {
            "protocol_fingerprint": "fingerprint",
            "status": "ok",
            "track_id": "b",
            "query_seconds": 2.0,
            "condition": "clean",
            "correct_rank": 7,
            "latency_seconds": 1.0,
        },
        {
            "protocol_fingerprint": "fingerprint",
            "status": "error",
            "track_id": "c",
            "query_seconds": 2.0,
            "condition": "clean",
            "latency_seconds": 0.5,
        },
    ]
    path = tmp_path / "queries.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    restored = _load_joint_rows(path, fingerprint="fingerprint")
    assert len(restored) == 3
    metrics = _joint_metrics(list(restored.values()), selected_tracks=3)
    assert metrics["evaluated_queries"] == 2
    assert metrics["failed_queries"] == 1
    assert metrics["beam_top1"] == 1 / 3
    assert metrics["beam_top5"] == 1 / 3
    assert metrics["beam_top10"] == 2 / 3
    assert metrics["beam_mrr"] == pytest.approx((1 + 1 / 7) / 3)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        _load_joint_rows(path, fingerprint="other")


def test_room_ir_assets_reject_shared_room_or_identical_content(tmp_path):
    train = tmp_path / "train" / "OpenAIR" / "room-a"
    validation = tmp_path / "validation" / "OpenAIR" / "room-a"
    train.mkdir(parents=True)
    validation.mkdir(parents=True)
    impulse = np.zeros(16, dtype=np.float32)
    impulse[0] = 1.0
    sf.write(train / "train.wav", impulse, 8_000)
    sf.write(validation / "test.wav", impulse, 8_000)
    with pytest.raises(ValueError, match="overlap"):
        RoomImpulseResponseAssets(
            train.parents[1], validation.parents[1], sample_rate=8_000
        )


def test_monitor_collator_skips_invalid_crop_without_aborting(tmp_path):
    audio_root = tmp_path / "audio"
    train_noise = tmp_path / "noise_train"
    validation_noise = tmp_path / "noise_validation"
    audio_root.mkdir()
    train_noise.mkdir()
    validation_noise.mkdir()
    rir_train = tmp_path / "rir_train" / "OpenAIR" / "train-room"
    rir_validation = tmp_path / "rir_validation" / "OpenAIR" / "test-room"
    rir_train.mkdir(parents=True)
    rir_validation.mkdir(parents=True)
    waveform = np.sin(
        2 * np.pi * 220 * np.arange(6 * 8_000, dtype=np.float32) / 8_000
    )
    sf.write(audio_root / "valid.wav", waveform, 8_000)
    sf.write(train_noise / "train.wav", waveform[:40_000], 8_000)
    sf.write(validation_noise / "validation.wav", waveform[:40_000], 8_000)
    impulse = np.zeros(800, dtype=np.float32)
    impulse[0] = 1.0
    sf.write(rir_train / "ir.wav", impulse, 8_000)
    sf.write(rir_validation / "ir.wav", impulse[::-1], 8_000)
    assets = BackgroundNoiseAssets(
        train_noise, validation_noise, sample_rate=8_000, samples=16_000
    )
    rir_assets = RoomImpulseResponseAssets(
        rir_train.parent.parent, rir_validation.parent.parent, sample_rate=8_000
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
        rir_assets=rir_assets,
        sample_rate=8_000,
        past_context_duration=2.0,
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
