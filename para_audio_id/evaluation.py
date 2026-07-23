from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .audio import BadFileRegistry, load_audio, quantile_normalize
from .augment import WaveformAugmenter
from .catalogue import CatalogueRecord, load_catalogue
from .checkpoint import load_network
from .codes import code_to_tokens
from .metrics import ranking_metrics


def evaluation_queries(
    cfg: dict,
    records: list[CatalogueRecord],
    *,
    degraded: bool,
    max_queries: int | None,
):
    root = Path(cfg["data"]["audio_root"])
    duration = float(cfg["data"]["query_duration"])
    sample_rate = int(cfg["model"]["sample_rate"])
    registry = BadFileRegistry(cfg["data"]["runtime_bad_files"])
    augmenter = (
        WaveformAugmenter(cfg["data"]["augmentation"], sample_rate, cfg["train"]["seed"] + 991)
        if degraded
        else None
    )
    emitted = 0
    for record_index, record in enumerate(records):
        for position in cfg["evaluation"]["positions"]:
            if max_queries is not None and emitted >= max_queries:
                return
            if registry.contains(record.path):
                continue
            start = max(0.0, (record.duration - duration) * float(position))
            try:
                audio = load_audio(
                    root / record.path,
                    sample_rate=sample_rate,
                    start=start,
                    duration=duration,
                    pad=True,
                )
                if not np.isfinite(audio).all():
                    raise ValueError("non-finite decoded samples")
            except Exception as exc:
                registry.add(record.path, exc)
                continue
            augmentation = {}
            if augmenter is not None:
                audio, augmentation = augmenter(audio)
            audio = quantile_normalize(audio, float(cfg["model"]["quantile_norm"]))
            emitted += 1
            yield {
                "audio": torch.from_numpy(audio),
                "target": code_to_tokens(record.code),
                "code": record.code,
                "path": record.path,
                "position": float(position),
                "start": start,
                "augmentation": augmentation,
            }


def evaluate(
    checkpoint_path: str | Path,
    *,
    output: str | Path,
    degraded: bool = False,
    max_queries: int | None = None,
    device: str = "cuda",
    beam_width: int = 10,
) -> dict:
    network, cfg, checkpoint = load_network(checkpoint_path, device)
    records = (
        [CatalogueRecord(**record) for record in checkpoint["catalogue"]]
        if "catalogue" in checkpoint
        else load_catalogue(cfg["data"]["catalogue"])
    )
    targets: list[str] = []
    greedy: list[str] = []
    rankings = []
    digit_correct = 0
    digit_total = 0
    latency = 0.0
    rows = []
    for query in evaluation_queries(
        cfg, records, degraded=degraded, max_queries=max_queries
    ):
        audio = query["audio"].unsqueeze(0).to(device)
        target = query["target"].unsqueeze(0).to(device)
        started = time.perf_counter()
        with torch.inference_mode():
            logits = network(audio, target)
            greedy_code = network.greedy_decode(audio)[0]
            ranking = network.beam_decode(audio, width=beam_width)[0]
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        latency += time.perf_counter() - started
        digit_correct += int((logits.argmax(-1) == target).sum())
        digit_total += target.numel()
        targets.append(query["code"])
        greedy.append(greedy_code)
        rankings.append(ranking)
        rows.append(
            {
                **{key: query[key] for key in ("code", "path", "position", "start", "augmentation")},
                "greedy": greedy_code,
                "beam": [
                    {"code": item.code, "log_probability": item.log_probability}
                    for item in ranking
                ],
            }
        )
    if not targets:
        raise RuntimeError("No evaluation queries could be decoded")
    metrics = ranking_metrics(targets, rankings)
    metrics.update(
        {
            "greedy_top1": sum(a == b for a, b in zip(targets, greedy, strict=True))
            / len(targets),
            "teacher_forced_digit_accuracy": digit_correct / digit_total,
            "queries": len(targets),
            "mean_latency_seconds": latency / len(targets),
            "degraded": degraded,
        }
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"metrics": metrics, "queries": rows}, indent=2))
    return metrics
