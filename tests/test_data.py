import pytest

from para_audio_id.catalogue import CatalogueRecord
from para_audio_id.data import (
    CatalogueSegmentDataset,
    IdentityGroupedBatchSampler,
    build_segment_inventory,
    canonical_starts,
)


def records(count: int, duration: float = 30.0) -> list[CatalogueRecord]:
    return [
        CatalogueRecord(
            path=f"{index}.mp3",
            track_id=f"track-{index}",
            code=f"{index:05d}",
            duration=duration,
        )
        for index in range(count)
    ]


def test_canonical_starts_cover_aligned_non_aligned_and_minimum_duration():
    assert canonical_starts(30.0, 5.0, 5.0) == (0.0, 5.0, 10.0, 15.0, 20.0, 25.0)
    assert canonical_starts(29.0, 5.0, 5.0) == (0.0, 5.0, 10.0, 15.0, 20.0, 24.0)
    assert canonical_starts(5.0, 5.0, 5.0) == (0.0,)
    with pytest.raises(ValueError, match="shorter"):
        canonical_starts(4.9, 5.0, 5.0)


def test_inventory_covers_every_record_deterministically():
    first, first_by_record = build_segment_inventory(records(3), 5.0, 5.0)
    second, second_by_record = build_segment_inventory(records(3), 5.0, 5.0)
    assert first == second
    assert first_by_record == second_by_record
    assert len(first) == 18
    assert {segment.record_index for segment in first} == {0, 1, 2}


class FakeDataset:
    def __init__(self, count=16):
        self.segments_by_record = [
            list(range(record * 6, (record + 1) * 6)) for record in range(count)
        ]
        self.valid = list(range(count))

    def valid_record_indices(self):
        return self.valid


def test_grouped_sampler_has_eight_identities_and_eight_views():
    dataset = FakeDataset()
    sampler = IdentityGroupedBatchSampler(
        dataset, songs_per_batch=8, views_per_song=8, seed=7
    )
    batches = list(sampler)
    assert len(batches) == 2
    for batch in batches:
        identities = [index // 6 for index in batch]
        assert len(batch) == 64
        assert len(set(identities)) == 8
        assert all(identities.count(identity) == 8 for identity in set(identities))


def test_grouped_sampler_is_rank_stable_and_rejects_invalid_global_batch():
    dataset = FakeDataset()
    rank_zero = IdentityGroupedBatchSampler(
        dataset, songs_per_batch=8, views_per_song=8, seed=7, world_size=2, rank=0
    )
    rank_one = IdentityGroupedBatchSampler(
        dataset, songs_per_batch=8, views_per_song=8, seed=7, world_size=2, rank=1
    )
    for left, right in zip(rank_zero, rank_one, strict=True):
        assert len(left) == len(right) == 32
        assert {index // 6 for index in left}.isdisjoint(
            {index // 6 for index in right}
        )
    with pytest.raises(ValueError, match="must divide"):
        IdentityGroupedBatchSampler(
            dataset, songs_per_batch=8, views_per_song=8, seed=7, world_size=3
        )


def test_grouped_sampler_exposure_order_is_reproducible():
    dataset = FakeDataset()
    first = IdentityGroupedBatchSampler(
        dataset, songs_per_batch=8, views_per_song=8, seed=7, exposure=3
    )
    resumed = IdentityGroupedBatchSampler(
        dataset, songs_per_batch=8, views_per_song=8, seed=7, exposure=3
    )
    next_exposure = IdentityGroupedBatchSampler(
        dataset, songs_per_batch=8, views_per_song=8, seed=7, exposure=4
    )
    assert list(first) == list(resumed)
    assert list(resumed) != list(next_exposure)


def test_disabled_augmentation_does_not_load_assets(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "para_audio_id.data.load_catalogue", lambda _: records(1)
    )
    augmentation = {
        name: {
            "enabled": False,
            "probability": 0.0,
            "root": str(tmp_path / "missing"),
        }
        for name in (
            "pitch_shift",
            "time_stretch",
            "resampling",
            "background",
            "room_ir",
            "microphone_ir",
        )
    }
    cfg = {
        "model": {"sample_rate": 24_000, "quantile_norm": 0.95},
        "data": {
            "audio_root": str(tmp_path),
            "catalogue": str(tmp_path / "catalogue.jsonl"),
            "runtime_bad_files": str(tmp_path / "bad.jsonl"),
            "query_duration": 5.0,
            "segment_stride": 5.0,
            "augmentation": augmentation,
        },
        "train": {"seed": 7},
    }
    dataset = CatalogueSegmentDataset(cfg, training=True)
    assert dataset.augmenter is None
