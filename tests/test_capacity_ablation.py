from __future__ import annotations

import json
from pathlib import Path

import pytest

from para_audio_id.audio_lm.capacity_ablation import (
    CAPACITY_ABLATION_AUDIO_TOKENS,
    CAPACITY_ABLATION_TRACKS,
    aggregate_capacity_queries,
    _validate_capacity_checkpoint,
    capacity_ablation_paths,
    capacity_run_name,
    reciprocal_rank,
    validate_nested_capacity_cohorts,
)
from para_audio_id.audio_lm.profiles import (
    canonical_capacity_profile,
    catalogue_fingerprint,
)
from para_audio_id.audio_lm.vocabulary import AudioLMVocabulary
from para_audio_id.catalogue import CatalogueRecord


def _write_manifest(path: Path, records: list[CatalogueRecord], ids: list[str]) -> None:
    selected = {record.track_id: record for record in records}
    import hashlib

    payload = {
        "protocol": "fresh_seeded_catalogue_cohort_v1",
        "seed": 1337,
        "database_size": len(ids),
        "count": len(ids),
        "catalogue_fingerprint": catalogue_fingerprint(records),
        "track_ids": ids,
        "code_mapping_fingerprint": hashlib.sha256(
            "\n".join(f"{track_id}:{selected[track_id].code}" for track_id in ids).encode()
        ).hexdigest(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_capacity_paths_and_rank_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = {"train": {"checkpoint_dir": "/checkpoints"}}
    checkpoint, result, manifest = capacity_ablation_paths(cfg, 25_000, "small")
    assert checkpoint == Path("/checkpoints/25k-small-cb8/last.ckpt")
    assert result == Path("capacity-ablation-results/25k-small.json")
    assert manifest.name == "clean-2s-common-1k.manifest.json"
    assert capacity_run_name(100_000, "medium") == "100k-medium-cb8"
    assert CAPACITY_ABLATION_AUDIO_TOKENS == 400
    assert reciprocal_rank(["00001", "00002"], "00002") == 0.5
    assert reciprocal_rank(["00001"], "99999") == 0.0


def test_capacity_metrics_use_all_selected_queries() -> None:
    rows = [
        {"status": "ok", "correct_rank": 1, "reciprocal_rank": 1.0},
        {"status": "ok", "correct_rank": 7, "reciprocal_rank": 1 / 7},
        {"status": "error", "correct_rank": None, "reciprocal_rank": 0.0},
    ]
    metrics = aggregate_capacity_queries(rows, 3)
    assert metrics["beam_mrr"] == pytest.approx((1 + 1 / 7) / 3)
    assert metrics["beam_top1"] == pytest.approx(1 / 3)
    assert metrics["beam_top5"] == pytest.approx(1 / 3)
    assert metrics["beam_top10"] == pytest.approx(2 / 3)
    assert metrics["failed_queries"] == 1


def test_nested_capacity_validation_rejects_non_nested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sizes = (10_000, 25_000, 50_000, 100_000)
    records = [
        CatalogueRecord(
            path=f"{index}.mp3",
            track_id=f"track-{index}",
            code=f"{index:05d}",
            duration=30.0,
        )
        for index in range(100_000)
    ]
    monkeypatch.chdir(tmp_path)
    for size in sizes:
        ids = [record.track_id for record in records[:size]]
        _write_manifest(Path(f"data/training_tracks_{size // 1000}k.json"), records, ids)
    cohorts, _ = validate_nested_capacity_cohorts({}, records)
    assert len(cohorts[10_000]) == 10_000

    bad_ids = [record.track_id for record in records[10_000:35_000]]
    _write_manifest(Path("data/training_tracks_25k.json"), records, bad_ids)
    with pytest.raises(ValueError, match="are not nested"):
        validate_nested_capacity_cohorts({}, records)


def test_expected_query_denominator_is_fixed() -> None:
    assert CAPACITY_ABLATION_TRACKS == 1_000
    with pytest.raises(ValueError, match="one result"):
        aggregate_capacity_queries([], CAPACITY_ABLATION_TRACKS)


def test_capacity_checkpoint_profile_validation() -> None:
    track_ids = [f"track-{index}" for index in range(10_000)]
    profile = canonical_capacity_profile(
        database_size=10_000,
        decoder="tiny",
        target_exposures=560,
        tracks_per_optimizer_step=80,
    )
    checkpoint = {
        "architecture": "audio_lm_v1",
        "tokenizer_spec": {
            "selected_codebooks": 8,
        },
        "tokenizer_fingerprint": "tokenizer",
        "vocabulary": AudioLMVocabulary(8).to_dict(),
        "model_config": {},
        "code_mapping_fingerprint": "codes",
        "validation_probe": [],
        "training_track_ids": track_ids,
        "resolved_training_profile": profile,
    }
    assert _validate_capacity_checkpoint(
        checkpoint,
        database_size=10_000,
        decoder="tiny",
        expected_track_ids=track_ids,
    ) == profile
    with pytest.raises(ValueError, match="decoder"):
        _validate_capacity_checkpoint(
            checkpoint,
            database_size=10_000,
            decoder="small",
            expected_track_ids=track_ids,
        )
    with pytest.raises(ValueError, match="identities"):
        _validate_capacity_checkpoint(
            checkpoint,
            database_size=10_000,
            decoder="tiny",
            expected_track_ids=list(reversed(track_ids)),
        )
