#!/usr/bin/env python3
"""Larger presentation version; leaves the paper figure unchanged."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, MultipleLocator, NullFormatter

from capacity import COLORS, DEFAULT_RUNS, RECORDINGS_PER_UPDATE, database_size_from_run, load_curve


def main():
    root = Path(__file__).resolve().parents[1]
    plt.rcParams.update({
        "font.size": 12, "axes.labelsize": 14, "legend.fontsize": 12,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for run, color in zip(DEFAULT_RUNS, COLORS):
        steps, values = load_curve(
            root / "logs/fma-large-audio-lm" / run / "training_metrics.jsonl",
            "train/digit_loss_step",
        )
        size = database_size_from_run(run)
        exposure = [step * RECORDINGS_PER_UPDATE / size for step in steps]
        for ax, x in zip(axes, (steps, exposure)):
            ax.plot(x, values, color=color, linewidth=1.2, label=f"{size // 1000}K")

    axes[0].set_ylabel("Identifier digit cross-entropy")
    axes[0].set_xlabel("Optimiser updates (log scale)")
    axes[1].set_xlabel(r"Per-recording exposure $UB/N$ (log scale)")
    for ax in axes:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10), numticks=100))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.set_ylim(-0.05, 4.4)
        ax.grid(which="major", alpha=0.23, linewidth=0.7)
        ax.grid(which="minor", axis="x", alpha=0.08, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.axhline(math.log(10), color="#555555", linestyle=":", linewidth=1.3)
        ax.tick_params(which="major", length=5)
        ax.tick_params(which="minor", length=2.5)
    for level, label in ((math.log(10), r"$\ell_0=\ln(10)$"),):
        axes[0].annotate(label, xy=(0.98, level),
                         xycoords=axes[0].get_yaxis_transform(),
                         xytext=(0, 4), textcoords="offset points",
                         ha="right", va="bottom", fontsize=11,
                         bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="DB Size", ncol=4, frameon=False,
               loc="upper center", bbox_to_anchor=(0.54, 1.0))
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.15, top=0.84, wspace=0.12)
    output = Path(__file__).with_name("capacity-expanded")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
