#!/usr/bin/env python3
"""Plot saved capacity-training curves against updates and exposure."""

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
RECORDINGS_PER_UPDATE = 80


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


def database_size_from_run(run: str) -> int:
    label = run.split("-", maxsplit=1)[0].lower()
    if not label.endswith("k") or not label[:-1].isdigit():
        raise ValueError(
            f"Cannot derive database size from run {run!r}; expected e.g. 25k-..."
        )
    return int(label[:-1]) * 1_000


def transition_onset(
    values: list[float], *, persistence: int, threshold: float
) -> int | None:
    """Return the first index below threshold for it and the next K values."""
    if persistence < 0:
        raise ValueError("crossing persistence must be non-negative")
    required = persistence + 1
    for index in range(len(values) - required + 1):
        if all(value < threshold for value in values[index : index + required]):
            return index
    return None


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Plot saved capacity loss curves against updates and exposure."
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
        default=25,
        help="trailing moving-average window in saved log records (default: 25)",
    )
    parser.add_argument(
        "--crossing-persistence",
        type=int,
        default=5,
        help="subsequent below-threshold measurements required (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output PDF (default: ablations/capacity.pdf)",
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
    if args.crossing_persistence < 0:
        raise SystemExit("--crossing-persistence must be non-negative")

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    updates_axis, exposures_axis = axes
    chance_loss = math.log(10)
    crossing_threshold = 0.9 * chance_loss
    for index, run in enumerate(args.runs):
        path = args.log_root / run / "training_metrics.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing saved training log: {path}")
        steps, values = load_curve(path, args.metric)
        values = moving_average(values, args.smoothing_window)
        database_size = database_size_from_run(run)
        exposures = [
            step * RECORDINGS_PER_UPDATE / database_size for step in steps
        ]
        color = COLORS[index % len(COLORS)]
        database_label = f"{database_size // 1_000}K"
        updates_axis.plot(
            steps,
            values,
            label=database_label,
            color=color,
            linewidth=1.4,
        )
        exposures_axis.plot(
            exposures,
            values,
            label=database_label,
            color=color,
            linewidth=1.4,
        )
        crossing_index = transition_onset(
            values,
            persistence=args.crossing_persistence,
            threshold=crossing_threshold,
        )
        if crossing_index is not None:
            crossing_message = (
                f"first persistent crossing of 0.9 ell_0: "
                f"step {steps[crossing_index]:,}, "
                f"exposure {exposures[crossing_index]:.3f}, "
                f"smoothed loss {values[crossing_index]:.6f}"
            )
        else:
            crossing_message = "0.9 ell_0 not reached"
        print(
            f"{run}: {len(steps)} points, steps {steps[0]:,}–{steps[-1]:,}; "
            f"{crossing_message}"
        )

    updates_axis.set_xlabel("Optimizer update (log scale)")
    exposures_axis.set_xlabel(r"Per-recording exposure $U B / N$ (log scale)")
    updates_axis.set_ylabel("Identifier digit cross-entropy")
    for axis in axes:
        axis.set_xscale("log")
        axis.xaxis.set_major_locator(LogLocator(base=10))
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.axhline(
            chance_loss,
            color="#3f3f3f",
            linestyle=":",
            linewidth=1.5,
            zorder=0,
        )
        axis.axhline(
            crossing_threshold,
            color="#a5a5a5",
            linestyle=":",
            linewidth=1.25,
            zorder=0,
        )
        axis.grid(True, which="major", alpha=0.30)
        axis.grid(True, which="minor", axis="x", alpha=0.12)

    exposures_axis.text(
        0.99,
        chance_loss,
        r"$\ell_0=\ln(10)$",
        transform=exposures_axis.get_yaxis_transform(),
        ha="right",
        va="bottom",
        color="#3f3f3f",
    )
    exposures_axis.text(
        0.99,
        crossing_threshold,
        r"$0.9\ell_0$",
        transform=exposures_axis.get_yaxis_transform(),
        ha="right",
        va="bottom",
        color="#858585",
    )
    figure.legend(
        *updates_axis.get_legend_handles_labels(),
        loc="upper center",
        ncol=len(args.runs),
        frameon=False,
    )
    figure.tight_layout()
    figure.subplots_adjust(top=0.90, wspace=0.06)

    output = args.output or (Path(__file__).resolve().parent / "capacity.pdf")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
