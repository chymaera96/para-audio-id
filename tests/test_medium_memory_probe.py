from __future__ import annotations

import pytest

from probe_medium_memory import recommend_candidate, validate_candidates


def test_probe_candidates_must_be_unique_ascending_positive_integers():
    assert validate_candidates([10, 12, 16, 20]) == [10, 12, 16, 20]
    with pytest.raises(ValueError, match="duplicates"):
        validate_candidates([10, 12, 12])
    with pytest.raises(ValueError, match="ascending positive"):
        validate_candidates([12, 10])
    with pytest.raises(ValueError, match="ascending positive"):
        validate_candidates([0, 10])


def test_probe_recommends_fastest_candidate_with_ten_percent_headroom():
    results = [
        {
            "tracks_per_gpu": 20,
            "success": True,
            "peak_headroom_percent": 20.0,
            "documents_per_second": 20.0,
        },
        {
            "tracks_per_gpu": 24,
            "success": True,
            "peak_headroom_percent": 12.0,
            "documents_per_second": 25.0,
        },
        {
            "tracks_per_gpu": 28,
            "success": True,
            "peak_headroom_percent": 8.0,
            "documents_per_second": 30.0,
        },
    ]
    recommendation = recommend_candidate(
        results, minimum_headroom_percent=10.0
    )
    assert recommendation == {
        "selected_tracks_per_gpu": 24,
        "documents_per_gpu": 48,
        "world_size": 2,
        "accumulate_grad_batches": 1,
        "global_tracks_per_optimizer_step": 48,
        "global_documents_per_optimizer_step": 96,
        "target_track_selections": 72_000_000,
        "resolved_max_steps": 1_500_000,
        "minimum_peak_headroom_percent": 10.0,
        "selection_rule": (
            "highest measured documents_per_second among safe candidates"
        ),
    }


def test_probe_rejects_results_without_required_headroom():
    with pytest.raises(RuntimeError, match="headroom"):
        recommend_candidate(
            [
                {
                    "tracks_per_gpu": 40,
                    "success": True,
                    "peak_headroom_percent": 5.0,
                    "documents_per_second": 30.0,
                }
            ],
            minimum_headroom_percent=10.0,
        )
