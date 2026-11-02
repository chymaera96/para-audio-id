#!/usr/bin/env python3
"""Plot user-supplied 2-second probe endpoints from 2026-10-09.

Runs: tc12-stable-cb2, tc16, tc17_wo_kd, tc18 (all 225000 updates).
These are historical runs, not a strictly controlled codebook-only ablation.
"""

from pathlib import Path

import matplotlib.pyplot as plt


def main():
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    codebooks = [2, 4, 6, 8]
    rir = [17.057, 36.455, 56.522, 58.863]
    noise_rir = [8.8294, 20.736, 35.786, 40.201]
    noise_30 = [83.946, 92.642, 96.656, 97.324]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(codebooks, rir, "o-", color="#087e8b", label="RIR",
            linewidth=1.6, markersize=5)
    ax.plot(codebooks, noise_rir, "s-", color="#e58b20", label="Noise + RIR",
            linewidth=1.6, markersize=5)
    ax.plot(codebooks, noise_30, "^-", color="#8055c9", label="Noise (30 dB SNR)",
            linewidth=1.6, markersize=5)
    ax.set(xlabel="Codebooks", ylabel="Top-1 hit rate (%)",
           xticks=codebooks, yticks=list(range(0, 101, 5)), ylim=(0, 105), xlim=(1.7, 8.3))
    for values, color in ((rir, "#087e8b"), (noise_rir, "#e58b20"),
                          (noise_30, "#8055c9")):
        for x, value in zip(codebooks, values):
            ax.annotate(f"{value:.2f}%", (x, value),
                        xytext=(0, 9), textcoords="offset points",
                        ha="center", va="bottom", color=color, fontsize=10)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False)
    fig.tight_layout(pad=0.6)
    output = Path(__file__).with_suffix("")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
