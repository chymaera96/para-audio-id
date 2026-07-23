from __future__ import annotations

from collections.abc import Sequence

from .model import BeamResult


def ranking_metrics(
    targets: Sequence[str], rankings: Sequence[Sequence[BeamResult]]
) -> dict[str, float]:
    if len(targets) != len(rankings) or not targets:
        raise ValueError("Targets and non-empty rankings must have equal length")
    hits = {1: 0, 5: 0, 10: 0}
    reciprocal_rank = 0.0
    for target, ranking in zip(targets, rankings, strict=True):
        codes = [item.code for item in ranking]
        for k in hits:
            hits[k] += int(target in codes[:k])
        if target in codes:
            reciprocal_rank += 1.0 / (codes.index(target) + 1)
    count = len(targets)
    return {
        "beam_top1": hits[1] / count,
        "beam_top5": hits[5] / count,
        "beam_top10": hits[10] / count,
        "beam_mrr": reciprocal_rank / count,
    }
