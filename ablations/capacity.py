#!/usr/bin/env python3
"""Plot saved capacity-training curves with a logarithmic step axis."""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
from pathlib import Path
from typing import Iterable


DEFAULT_RUNS = (
    "100k-small-cb8",
    "50k-small-cb8",
    "25k-small-cb8",
    "10k-small-cb8",
)
COLORS = ("#ff4d57", "#8bc34a", "#159184", "#44b7cf")


def metric_json_key(name: str) -> str:
    """Map a W&B display name to its key in training_metrics.jsonl."""
    key = name.removeprefix("train/")
    for suffix in ("_step", "_epoch"):
        if key.endswith(suffix):
            key = key[: -len(suffix)]
            break
    return key


def load_curve(path: Path, metric: str) -> tuple[list[int], list[float]]:
    """Load the latest finite metric value at each positive optimizer step."""
    values_by_step: dict[int, float] = {}
    key = metric_json_key(metric)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            step = int(row["global_step"])
            value = row.get("metrics", {}).get(key)
            if step <= 0 or value is None:
                continue
            value = float(value)
            if math.isfinite(value):
                # Resumed jobs may append a second value for a previously seen step.
                # Keeping the latest mirrors the final state of the saved run.
                values_by_step[step] = value
    if not values_by_step:
        raise ValueError(f"No positive-step {metric!r} values found in {path}")
    points = sorted(values_by_step.items())
    return [step for step, _ in points], [value for _, value in points]


def moving_average(values: Iterable[float], window: int) -> list[float]:
    if window < 1:
        raise ValueError("smoothing window must be at least one")
    if window == 1:
        return list(values)
    queue: deque[float] = deque()
    total = 0.0
    smoothed: list[float] = []
    for value in values:
        queue.append(value)
        total += value
        if len(queue) > window:
            total -= queue.popleft()
        smoothed.append(total / len(queue))
    return smoothed


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Plot saved capacity loss curves on a log x-axis."
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=repository / "logs" / "fma-large-audio-lm",
        help="directory containing the run directories",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=list(DEFAULT_RUNS),
        help="run-directory names, in legend order",
    )
    parser.add_argument(
        "--metric",
        default="train/digit_loss_step",
        help="W&B-style metric name or JSONL metric key",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=1,
        help="trailing moving-average window in saved log records (default: raw)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repository / "ablations" / "capacity.pdf",
    )
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import LogLocator, NullFormatter
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required: python -m pip install matplotlib"
        ) from exc

    if args.smoothing_window < 1:
        raise SystemExit("--smoothing-window must be at least one")

    figure, axis = plt.subplots(figsize=(10, 6))
    for index, run in enumerate(args.runs):
        path = args.log_root / run / "training_metrics.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing saved training log: {path}")
        steps, values = load_curve(path, args.metric)
        values = moving_average(values, args.smoothing_window)
        color = COLORS[index % len(COLORS)]
        database_size = run.split("-", maxsplit=1)[0].upper()
        axis.plot(
            steps,
            values,
            label=database_size,
            color=color,
            linewidth=1.4,
        )
        print(
            f"{run}: {len(steps)} points, steps {steps[0]:,}–{steps[-1]:,}"
        )

    axis.set_xscale("log")
    axis.xaxis.set_major_locator(LogLocator(base=10))
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.set_xlabel("Optimizer step (log scale)")
    axis.set_ylabel("Identifier digit cross-entropy")
    chance_loss = math.log(10)
    axis.axhline(
        chance_loss,
        color="#555555",
        linestyle=":",
        linewidth=1.3,
        zorder=0,
    )
    axis.text(
        0.99,
        chance_loss,
        r"$\ln(10)=2.303$",
        transform=axis.get_yaxis_transform(),
        ha="right",
        va="bottom",
        color="#444444",
    )
    axis.grid(True, which="major", alpha=0.30)
    axis.grid(True, which="minor", axis="x", alpha=0.12)
    axis.legend(frameon=False)
    figure.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
