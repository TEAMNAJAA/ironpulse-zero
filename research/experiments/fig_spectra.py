import os, sys
import numpy as np
import matplotlib.pyplot as plt
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)
sys.path.insert(0, os.path.dirname(_PROJ))
from experiments.common import CFG, cwru_index_48k, maf_index_all
from core import pipeline, camera_sim as cs, dsp, figures

figures.use()
FIGDIR = CFG["paths"]["figures"]
EXP = CFG["camera"]["default_exposure_sec"]
MAX_ORDER = 12.0


def order_spectrum(sig, fs, f0, max_order):
    f, A = dsp.amplitude_spectrum(sig, fs, mode=CFG["signal"]["detrend"],
                                  win=CFG["signal"]["window_fn"])
    o = f / f0
    m = (o >= 0.15) & (o <= max_order)
    return o[m], A[m]


def nearest(index, label, target_f0):
    best, bf = None, None
    for r in index:
        if r["label"] != label:
            continue
        f0 = r.get("nominal_f0") or (r.get("rpm_doc") or 0) / 60.0
        if not f0:
            continue
        d = abs(f0 - target_f0)
        if bf is None or d < bf:
            best, bf = r, d
    return best


def panel(ax, rec, title):
    sig, fsb, m = pipeline.signal_for(rec, CFG, domain="displacement")
    if sig is None:
        return None
    f0 = m["f0_hz"]
    combos = [(fsb, "ideal", None, "48 kHz reference", 0),
              (240.0, "ideal", None, "240 Hz  ideal", 1),
              (240.0, "boxcar", EXP, "240 Hz  boxcar", 2),
              (240.0, "naive", None, "240 Hz  naive", 3)]
    for fs_out, mode, exp, lab, ci in combos:
        y = cs.decimate(sig, fsb, fs_out, mode=mode, exposure_sec=exp)
        y = cs.to_pixels(y, CFG["optics"]["pixels_per_meter"])
        o, A = order_spectrum(y, fs_out, f0, MAX_ORDER)
        if not len(o):
            continue
        st = figures.style(ci)
        ax.semilogy(o, np.maximum(A, 1e-9), color=st["color"], linestyle=st["linestyle"],
                    linewidth=2.6 if ci == 0 else 1.7,
                    alpha=0.6 if ci == 0 else 0.95, label=lab)
    nyq = 120.0 / f0
    lo, hi = ax.get_ylim()
    if nyq <= MAX_ORDER:
        ax.axvspan(nyq, MAX_ORDER, color="#bbbbbb", alpha=0.35, zorder=0)
        ax.axvline(nyq, color="#555555", linestyle=":", linewidth=2.4)
        ax.text(min(nyq + 0.15, MAX_ORDER - 0.1), lo * 3.0,
                "lost at 240 Hz\n(above Nyquist, order %.1f)" % nyq,
                fontsize=12, color="#333333", va="bottom")
    for k in [1, 2, 3]:
        if k <= MAX_ORDER:
            ax.axvline(k, color="#cccccc", linewidth=1.2, zorder=0)
            ax.text(k, hi * 0.55, "%dx" % k, fontsize=12, ha="center", color="#777777")
    ax.set_xlim(0.15, MAX_ORDER)
    ax.set_title("%s        shaft speed f0 = %.1f Hz  (%.0f rpm)" %
                 (title, f0, f0 * 60), fontsize=15)
    ax.set_ylabel("amplitude (px)")
    return f0


def make_speed_story():
    idx = maf_index_all()
    targets = [("normal", 12.3, "normal, slow"),
               ("imbalance", 12.3, "imbalance, slow"),
               ("imbalance", 30.0, "imbalance, medium"),
               ("imbalance", 60.0, "imbalance, fast")]
    recs = [(nearest(idx, lab, f), t) for lab, f, t in targets]
    recs = [(r, t) for r, t in recs if r]
    if not recs:
        return
    fig, axes = plt.subplots(len(recs), 1, figsize=(11.5, 3.5 * len(recs)), sharex=True)
    if len(recs) == 1:
        axes = [axes]
    for ax, (r, t) in zip(axes, recs):
        panel(ax, r, t)
    axes[-1].set_xlabel("Order (multiples of shaft speed)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=12, frameon=False,
               bbox_to_anchor=(0.5, -0.012))
    fig.suptitle("Why shaft speed decides whether 240 Hz is enough\n"
                 "MAFAULDA, displacement domain; grey = destroyed by the 240 Hz Nyquist limit",
                 fontsize=17, y=0.999)
    figures.save(fig, os.path.join(FIGDIR, "fig_spectra_speed_story.png"))


def make_fault_types():
    idx = maf_index_all()
    pairs = [("normal", "normal"),
             ("imbalance", "imbalance"),
             ("horizontal_misalignment", "horizontal misalignment"),
             ("underhang_outer_race", "bearing outer race")]
    recs = []
    for lab, t in pairs:
        r = nearest(idx, lab, 25.0)
        if r:
            recs.append((r, t))
    if not recs:
        return
    fig, axes = plt.subplots(len(recs), 1, figsize=(11.5, 3.5 * len(recs)), sharex=True)
    if len(recs) == 1:
        axes = [axes]
    for ax, (r, t) in zip(axes, recs):
        panel(ax, r, t)
    axes[-1].set_xlabel("Order (multiples of shaft speed)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=12, frameon=False,
               bbox_to_anchor=(0.5, -0.012))
    fig.suptitle("Fault signatures in the displacement domain at a matched speed\n"
                 "MAFAULDA; grey = destroyed by the 240 Hz Nyquist limit", fontsize=17, y=0.999)
    figures.save(fig, os.path.join(FIGDIR, "fig_spectra_fault_types.png"))


def make_cwru():
    idx = cwru_index_48k()
    pairs = [("normal", "normal"), ("inner_race", "bearing inner race"),
             ("outer_race", "bearing outer race"), ("ball", "bearing ball")]
    recs = []
    for lab, t in pairs:
        sel = [r for r in idx if r["label"] == lab]
        if sel:
            recs.append((sel[0], t))
    if not recs:
        return
    fig, axes = plt.subplots(len(recs), 1, figsize=(11.5, 3.5 * len(recs)), sharex=True)
    if len(recs) == 1:
        axes = [axes]
    for ax, (r, t) in zip(axes, recs):
        panel(ax, r, t)
    axes[-1].set_xlabel("Order (multiples of shaft speed)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=12, frameon=False,
               bbox_to_anchor=(0.5, -0.012))
    fig.suptitle("CWRU bearing faults in the displacement domain\n"
                 "grey = destroyed by the 240 Hz Nyquist limit", fontsize=17, y=0.999)
    figures.save(fig, os.path.join(FIGDIR, "fig_spectra_cwru.png"))


if __name__ == "__main__":
    for fn in [make_speed_story, make_fault_types, make_cwru]:
        try:
            fn()
        except Exception as e:
            print("skip", fn.__name__, type(e).__name__, e)
    print("spectra done")
