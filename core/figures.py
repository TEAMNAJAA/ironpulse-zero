import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OKABE_ITO = ["#000000", "#E69F00", "#56B4E9", "#009E73",
             "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1))]

BASE = {
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.labelsize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "axes.linewidth": 1.4,
    "lines.linewidth": 2.6,
    "lines.markersize": 8,
    "xtick.major.width": 1.4,
    "ytick.major.width": 1.4,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.8,
    "figure.autolayout": False,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
}


def use():
    plt.rcParams.update(BASE)


def style(i):
    return dict(color=OKABE_ITO[i % len(OKABE_ITO)],
                marker=MARKERS[i % len(MARKERS)],
                linestyle=LINESTYLES[i % len(LINESTYLES)])


def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("figure ->", path)


def rate_axis(ax, rates, primary=240):
    ax.set_xscale("log")
    ax.set_xticks(rates)
    ax.set_xticklabels([str(int(r)) for r in rates])
    ax.minorticks_off()
    if primary:
        ax.axvline(primary, color="#888888", linestyle=":", linewidth=2.0, zorder=0)
    ax.set_xlabel("Sampling rate (Hz)")


def chance_line(ax):
    ax.axhline(0.5, color="#666666", linewidth=1.4, linestyle=(0, (2, 2)), zorder=0)
