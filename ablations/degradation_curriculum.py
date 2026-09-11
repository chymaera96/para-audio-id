"""Plot the database-size-scaled degradation curriculum."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REFERENCE_DATABASE_SIZE = 25_000
# Display coordinates reproduce the deliberately compressed constant tail in the
# original figure. The final coordinate is a visual endpoint, not a linear step.
DISPLAY_STEPS = np.array([0.0, 10.0, 30.0, 60.0, 115.0])


def plot_curriculum(database_size: int, output: Path) -> None:
    if database_size <= 0:
        raise ValueError("database_size must be positive")

    scale = database_size / REFERENCE_DATABASE_SIZE
    clean = np.array([1.0, 1.0, 0.40, 0.10, 0.10])
    noise = np.array([0.0, 0.0, 0.30, 0.35, 0.35])
    rir = np.array([0.0, 0.0, 0.30, 0.30, 0.30])
    noise_rir = np.array([0.0, 0.0, 0.0, 0.25, 0.25])

    plt.rcParams.update({"font.size": 8, "axes.labelsize": 9})
    figure, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(3.323, 2.897),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.08},
    )

    top.stackplot(
        DISPLAY_STEPS,
        clean,
        noise,
        rir,
        noise_rir,
        labels=("Clean", "Noise", "RIR", "Noise + RIR"),
        colors=("#b7b7b7", "#157db5", "#efa900", "#0c9f78"),
    )
    top.set_ylim(0.0, 1.0)
    top.set_yticks([0.0, 0.5, 1.0])
    top.set_ylabel("Pair probability")
    top.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        fontsize=7,
        columnspacing=1.0,
        handlelength=1.4,
    )

    rir_steps = np.array([10.0, 30.0, 60.0, 115.0])
    eligible = np.array([1 / 3, 2 / 3, 1.0, 1.0])
    bottom.fill_between(rir_steps, eligible, color="#d95f02", alpha=0.13)
    bottom.plot(rir_steps, eligible, color="#d95f02", linewidth=2.0)
    bottom.set_ylim(0.0, 1.08)
    bottom.set_yticks([1 / 3, 2 / 3, 1.0], labels=["1/3", "2/3", "1"])
    bottom.set_ylabel("Eligible RIRs")
    bottom.set_xlabel("Optimiser updates (thousands)")

    # Preserve the original display positions and change only their values. The
    # ellipsis marks the compressed 400K--900K constant portion.
    tick_positions = [0.0, 10.0, 30.0, 60.0, 100.0, 107.5, 115.0]
    reference_labels = [0.0, 10.0, 30.0, 60.0, 100.0]
    tick_labels = [f"{value * scale:g}" for value in reference_labels]
    tick_labels.extend(["⋯", f"{225.0 * scale:g}"])
    bottom.set_xticks(tick_positions, labels=tick_labels)
    bottom.get_xticklabels()[0].set_horizontalalignment("right")
    bottom.set_xlim(0.0, DISPLAY_STEPS[-1])

    for axis in (top, bottom):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(direction="out", length=4, width=0.8)
    bottom.grid(axis="y", color="#d0d0d0", linewidth=0.5, alpha=0.8)

    # Match the original canvas and axes placement; only the tick values differ.
    figure.subplots_adjust(left=0.14, right=0.96, bottom=0.17, top=0.87)

    output.parent.mkdir(parents=True, exist_ok=True)
    # Preserve the original Matplotlib figure canvas exactly. Tight bounding-box
    # export changes the apparent font and spacing at a fixed LaTeX width.
    figure.savefig(output)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-size", type=int, default=100_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("degradation_curriculum_100k.pdf"),
    )
    args = parser.parse_args()
    plot_curriculum(args.database_size, args.output)


if __name__ == "__main__":
    main()
