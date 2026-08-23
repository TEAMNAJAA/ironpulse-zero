import os, sys, csv
import numpy as np
import matplotlib.pyplot as plt
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)
sys.path.insert(0, os.path.dirname(_PROJ))
from experiments.common import CFG
from core import figures

figures.use()
T = CFG["paths"]["tables"]
FIGDIR = CFG["paths"]["figures"]
RATES = [float(r) for r in CFG["camera"]["target_fs_hz"]]
MODE_LABEL = {"ideal": "ideal  (anti-alias filter, lab instrument)",
              "naive": "naive  (no filter, full aliasing)",
              "boxcar": "boxcar  (real shutter, 1/240 s exposure)"}
FAULT_LABEL = {
    "imbalance": "imbalance",
    "horizontal_misalignment": "horiz. misalignment",
    "vertical_misalignment": "vert. misalignment",
    "underhang_cage_fault": "bearing cage (underhang)",
    "underhang_outer_race": "bearing outer race (underhang)",
    "underhang_ball_fault": "bearing ball (underhang)",
    "overhang_cage_fault": "bearing cage (overhang)",
    "overhang_outer_race": "bearing outer race (overhang)",
    "overhang_ball_fault": "bearing ball (overhang)",
    "inner_race": "bearing inner race",
    "outer_race": "bearing outer race",
    "ball": "bearing ball",
}


def load(name):
    p = os.path.join(T, name)
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return None
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    return rows or None


def fnum(r, k, d=np.nan):
    try:
        v = r.get(k, "")
        return float(v) if v not in ("", None) else d
    except (TypeError, ValueError):
        return d


def fig_e2_main(dataset="mafaulda", suffix=""):
    rows = load("e2_sampling_rate_overall%s.csv" % suffix)
    if not rows:
        return
    rows = [r for r in rows if r["dataset"] == dataset]
    if not rows:
        return
    full = [r for r in rows if fnum(r, "fs_hz") > 40000]
    ref = fnum(full[0], "roc_auc_mean") if full else np.nan
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, mode in enumerate(["ideal", "boxcar", "naive"]):
        sel = sorted([r for r in rows if r["mode"] == mode and fnum(r, "fs_hz") <= 40000],
                     key=lambda r: fnum(r, "fs_hz"))
        if not sel:
            continue
        x = [fnum(r, "fs_hz") for r in sel]
        y = [fnum(r, "roc_auc_mean") for r in sel]
        e = [fnum(r, "roc_auc_std") for r in sel]
        st = figures.style(i + 1)
        ax.errorbar(x, y, yerr=e, capsize=4, label=MODE_LABEL[mode], **st)
    if np.isfinite(ref):
        ax.axhline(ref, color="#000000", linewidth=2.0, linestyle="-", alpha=0.55)
        ax.text(min(RATES) * 1.04, ref + 0.014,
                "full 48 kHz reference = %.3f" % ref, fontsize=13, ha="left")
    figures.rate_axis(ax, RATES, primary=240)
    figures.chance_line(ax)
    ax.set_ylabel("ROC-AUC  (mean +/- s.d., 5-fold)")
    ax.set_title("E2  Does 240 Hz still separate faults?\n%s, displacement domain" %
                 dataset.upper())
    ax.set_ylim(0.4, 1.02)
    ax.legend(loc="lower right", framealpha=0.95)
    figures.save(fig, os.path.join(FIGDIR, "fig_e2_sampling_rate_%s.png" % dataset))


ROTOR = ["imbalance", "horizontal_misalignment", "vertical_misalignment"]


def fig_e2_per_fault(dataset="mafaulda", mode="boxcar", suffix=""):
    rows = load("e2_sampling_rate_per_label%s.csv" % suffix)
    if not rows:
        return
    rows = [r for r in rows if r["dataset"] == dataset and r["mode"] == mode]
    if not rows:
        return
    labs = sorted(set(r["label"] for r in rows))
    rotor = [l for l in labs if l in ROTOR]
    bearing = [l for l in labs if l not in ROTOR]
    btitle = ("bearing faults" if dataset != "mafaulda" else
              "bearing faults (all severities;" + chr(10) +
              " includes added rotor mass, see E1b)")
    groups = [(g, t) for g, t in [(rotor, "rotor-level faults"),
                                  (bearing, btitle)] if g]
    fig, axes = plt.subplots(1, len(groups), figsize=(7.2 * len(groups), 6.6),
                             sharey=True, squeeze=False)
    for ax, (grp, gtitle) in zip(axes[0], groups):
        for i, lab in enumerate(grp):
            sel = sorted([r for r in rows if r["label"] == lab and fnum(r, "fs_hz") <= 40000],
                         key=lambda r: fnum(r, "fs_hz"))
            if not sel:
                continue
            ax.plot([fnum(r, "fs_hz") for r in sel],
                    [fnum(r, "roc_auc_mean") for r in sel],
                    label=FAULT_LABEL.get(lab, lab), **figures.style(i))
        figures.rate_axis(ax, RATES, primary=240)
        figures.chance_line(ax)
        ax.set_ylim(0.42, 1.02)
        ax.set_title(gtitle, fontsize=14.5)
        ax.legend(loc="lower right", fontsize=11.5, framealpha=0.95)
    axes[0][0].set_ylabel("ROC-AUC")
    fig.suptitle("E2  Which fault type survives 240 Hz?  %s, displacement, %s shutter" %
                 (dataset.upper(), mode), fontsize=17, y=1.0)
    figures.save(fig, os.path.join(FIGDIR, "fig_e2_per_fault_%s.png" % dataset))


def fig_e1_domain():
    rows = load("e1_baseline_overall.csv")
    if not rows:
        return
    ds = sorted(set(r["dataset"] for r in rows))
    fig, ax = plt.subplots(figsize=(9, 6))
    w = 0.35
    xs = np.arange(len(ds))
    for i, dom in enumerate(["acceleration", "displacement"]):
        v, e = [], []
        for d in ds:
            m = [r for r in rows if r["dataset"] == d and r["domain"] == dom]
            v.append(fnum(m[0], "roc_auc_mean") if m else np.nan)
            e.append(fnum(m[0], "roc_auc_std") if m else 0.0)
        ax.bar(xs + (i - 0.5) * w, v, w, yerr=e, capsize=5,
               color=figures.OKABE_ITO[i + 1], edgecolor="black", linewidth=1.2,
               hatch=["", "//"][i],
               label="%s (%s)" % (dom, "accelerometer" if i == 0 else "what a camera sees"))
        for xx, vv, ee in zip(xs + (i - 0.5) * w, v, e):
            if np.isfinite(vv):
                ax.text(xx, vv + (ee if np.isfinite(ee) else 0) + 0.025, "%.3f" % vv,
                        ha="center", fontsize=13)
    figures.chance_line(ax)
    ax.set_xticks(xs)
    ax.set_xticklabels([d.upper() for d in ds])
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0, 1.12)
    ax.set_title("E1  The cost of measuring displacement instead of acceleration\n"
                 "both at the full 48 kHz rate")
    ax.legend(loc="lower left", framealpha=0.95)
    figures.save(fig, os.path.join(FIGDIR, "fig_e1_domain.png"))


def fig_e3():
    rows = load("e3_detectors_overall.csv")
    if not rows:
        return
    ds = sorted(set(r["dataset"] for r in rows))
    dets = sorted(set(r["detector"] for r in rows))
    fig, ax = plt.subplots(figsize=(10, 6))
    w = 0.8 / max(len(dets), 1)
    xs = np.arange(len(ds))
    for i, dt in enumerate(dets):
        v = [fnum(([r for r in rows if r["dataset"] == d and r["detector"] == dt] or [{}])[0],
                  "roc_auc_mean") for d in ds]
        e = [fnum(([r for r in rows if r["dataset"] == d and r["detector"] == dt] or [{}])[0],
                  "roc_auc_std", 0.0) for d in ds]
        ax.bar(xs + (i - (len(dets) - 1) / 2) * w, v, w, yerr=e, capsize=4,
               color=figures.OKABE_ITO[i + 1], edgecolor="black", linewidth=1.1,
               hatch=["", "//", "..", "xx"][i % 4], label=dt)
    figures.chance_line(ax)
    ax.set_xticks(xs)
    ax.set_xticklabels([d.upper() for d in ds])
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0, 1.05)
    ax.set_title("E3  Detector comparison at 240 Hz, boxcar shutter, displacement")
    ax.legend(loc="lower right", fontsize=12)
    figures.save(fig, os.path.join(FIGDIR, "fig_e3_detectors.png"))


def fig_e4():
    rows = load("e4_features.csv")
    if not rows:
        return
    for d in sorted(set(r["dataset"] for r in rows)):
        sel = [r for r in rows if r["dataset"] == d]
        order = ["all_dimensionless", "raw_units_only", "dimensionless_plus_raw"]
        drops = sorted([r["variant"] for r in sel if r["variant"].startswith("drop_")])
        onlys = sorted([r["variant"] for r in sel if r["variant"].startswith("only_")])
        names = order + drops + onlys
        vals, errs = [], []
        for n in names:
            m = [r for r in sel if r["variant"] == n]
            vals.append(fnum(m[0], "roc_auc_mean") if m else np.nan)
            errs.append(fnum(m[0], "roc_auc_std", 0.0) if m else 0.0)
        fig, ax = plt.subplots(figsize=(10, max(6, 0.45 * len(names) + 2)))
        y = np.arange(len(names))
        cols = [figures.OKABE_ITO[1] if n in order else
                (figures.OKABE_ITO[2] if n.startswith("drop_") else figures.OKABE_ITO[3])
                for n in names]
        ax.barh(y, vals, xerr=errs, capsize=4, color=cols, edgecolor="black", linewidth=1.1)
        ax.set_yticks(y)
        ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=12)
        ax.invert_yaxis()
        ax.axvline(0.5, color="#666666", linestyle=(0, (2, 2)), linewidth=1.4)
        base = vals[0] if vals else np.nan
        if np.isfinite(base):
            ax.axvline(base, color="black", linewidth=1.8, alpha=0.6)
        ax.set_xlabel("ROC-AUC")
        ax.set_xlim(0.3, 1.02)
        ax.set_title("E4  Feature ablation - %s\n240 Hz, boxcar, displacement" % d.upper())
        figures.save(fig, os.path.join(FIGDIR, "fig_e4_features_%s.png" % d))


SPLIT_LABEL = {"train_low_test_high": "slow -> fast",
               "train_high_test_low": "fast -> slow",
               "same_speed_reference": "same speed" + chr(10) + "(in-sample)",
               "cross_machine": "cross machine"}
DS_LABEL = {"cwru_48k": "CWRU", "mafaulda": "MAFAULDA",
            "train_mafaulda_test_cwru": "MAF -> CWRU"}
ORDERED = ["train_low_test_high", "train_high_test_low", "same_speed_reference",
           "cross_machine"]


def fig_e5():
    rows = load("e5_transfer.csv")
    if not rows:
        return
    groups = []
    for d in ["mafaulda", "cwru_48k", "train_mafaulda_test_cwru"]:
        for sp in ORDERED:
            if any(r["dataset"] == d and r["split"] == sp for r in rows):
                groups.append((d, sp))
    axes_kinds = ["order", "hz_fixed"]
    fig, axs = plt.subplots(1, 2, figsize=(17, 7.2))
    for panel, (metric, title, ylim) in enumerate(
            [("roc_auc", "ROC-AUC   (higher is better)", (0, 1.12)),
             ("far", "false-alarm rate   (LOWER is better)", (0, 1.18))]):
        ax = axs[panel]
        w = 0.8 / len(axes_kinds)
        xs = np.arange(len(groups))
        for i, ak in enumerate(axes_kinds):
            v = []
            for d, sp in groups:
                m = [r for r in rows if r["dataset"] == d and r["split"] == sp
                     and r["axis"] == ak]
                v.append(fnum(m[0], metric) if m else np.nan)
            lab = ("order axis (normalised by f0)" if ak == "order"
                   else "fixed Hz axis (no f0)")
            pos = xs + (i - (len(axes_kinds) - 1) / 2.0) * w
            ax.bar(pos, v, w, color=figures.OKABE_ITO[i + 3], edgecolor="black",
                   linewidth=1.1, hatch=["", "//"][i], label=lab)
            for xx, vv in zip(pos, v):
                if np.isfinite(vv):
                    ax.text(xx, vv + 0.02, "%.2f" % vv, ha="center", fontsize=11)
        if metric == "roc_auc":
            figures.chance_line(ax)
        ax.set_xticks(xs)
        ax.set_xticklabels([DS_LABEL.get(d, d) + chr(10) + SPLIT_LABEL.get(sp, sp)
                            for d, sp in groups], fontsize=10.5, rotation=30,
                           ha="right", rotation_mode="anchor")
        ax.set_ylabel(metric.replace("_", "-").upper())
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=15)
        ax.legend(loc="upper right", fontsize=11)
    fig.suptitle("E5  The order axis earns its keep in the FALSE-ALARM RATE, not the AUC" +
                 chr(10) + "240 Hz, boxcar, displacement", fontsize=17, y=1.01)
    figures.save(fig, os.path.join(FIGDIR, "fig_e5_transfer.png"))


def fig_e6():
    rows = load("e6_noise_overall.csv")
    if not rows:
        return
    for d in sorted(set(r["dataset"] for r in rows)):
        sel = [r for r in rows if r["dataset"] == d]
        clean = [r for r in sel if r["noise_type"] == "none"]
        fig, ax = plt.subplots(figsize=(9.5, 6.5))
        for i, nt in enumerate(["white", "shake"]):
            s = sorted([r for r in sel if r["noise_type"] == nt],
                       key=lambda r: -fnum(r, "snr_db"))
            if not s:
                continue
            lab = "white noise" if nt == "white" else "camera shake 8-12 Hz"
            ax.errorbar([fnum(r, "snr_db") for r in s],
                        [fnum(r, "roc_auc_mean") for r in s],
                        yerr=[fnum(r, "roc_auc_std", 0.0) for r in s],
                        capsize=4, label=lab, **figures.style(i + 1))
        if clean:
            c = fnum(clean[0], "roc_auc_mean")
            ax.axhline(c, color="black", linewidth=2.0, alpha=0.6)
            ax.text(29, c + 0.014, "noise-free = %.3f" % c, fontsize=13, ha="left")
        figures.chance_line(ax)
        ax.invert_xaxis()
        ax.set_xlabel("SNR (dB)   [lower = noisier]")
        ax.set_ylabel("ROC-AUC")
        ax.set_ylim(0.3, 1.03)
        ax.set_title("E6  Noise robustness - %s\n240 Hz, boxcar, displacement" % d.upper())
        ax.legend(loc="lower left")
        figures.save(fig, os.path.join(FIGDIR, "fig_e6_noise_%s.png" % d))


def fig_e7():
    rows = load("e7_resolution_overall.csv")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    for i, d in enumerate(sorted(set(r["dataset"] for r in rows))):
        sel = sorted([r for r in rows if r["dataset"] == d and fnum(r, "quant_px") > 0],
                     key=lambda r: fnum(r, "quant_px"))
        base = [r for r in rows if r["dataset"] == d and fnum(r, "quant_px") == 0]
        if not sel:
            continue
        ax.errorbar([fnum(r, "quant_px") for r in sel],
                    [fnum(r, "roc_auc_mean") for r in sel],
                    yerr=[fnum(r, "roc_auc_std", 0.0) for r in sel],
                    capsize=4, label=d.upper(), **figures.style(i + 1))
        if base:
            b = fnum(base[0], "roc_auc_mean")
            ax.axhline(b, color=figures.OKABE_ITO[i + 1], linewidth=1.6,
                       linestyle=(0, (1, 1)), alpha=0.8)
    ax.set_xscale("log")
    ax.set_xticks(CFG["camera"]["quantization_px"])
    ax.set_xticklabels([str(q) for q in CFG["camera"]["quantization_px"]])
    ax.minorticks_off()
    figures.chance_line(ax)
    ax.set_xlabel("Optical-flow resolution (pixels per step)")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.3, 1.03)
    ax.set_title("E7  How precise must optical flow be?\n"
                 "240 Hz, boxcar, displacement; dotted = no quantisation")
    ax.legend(loc="lower left")
    figures.save(fig, os.path.join(FIGDIR, "fig_e7_resolution.png"))


def fig_e8():
    rows = load("e8_trend_series.csv")
    summ = load("e8_trend_summary.csv")
    if not rows:
        return
    fsets = sorted(set(r.get("featureset", "dimensionless") for r in rows))
    for test in sorted(set(r["test"] for r in rows)):
        sel = [r for r in rows if r["test"] == test]
        conds = [c for c in ["accel_full", "accel_full_20kHz", "disp_full_48kHz",
                             "disp_240Hz_boxcar"]
                 if any(r["condition"] == c for r in sel)]
        if not conds or not fsets:
            continue
        fig, axes = plt.subplots(len(conds), len(fsets),
                                 figsize=(6.2 * len(fsets), 3.3 * len(conds)),
                                 sharex=True, squeeze=False)
        for i, c in enumerate(conds):
            for j, fsname in enumerate(fsets):
                ax = axes[i][j]
                s2 = sorted([r for r in sel if r["condition"] == c
                             and r.get("featureset", fsname) == fsname],
                            key=lambda r: fnum(r, "hours"))
                if not s2:
                    ax.set_visible(False)
                    continue
                h = [fnum(r, "hours") for r in s2]
                v = [fnum(r, "score") for r in s2]
                thr = fnum(s2[0], "threshold")
                ax.plot(h, v, color=figures.OKABE_ITO[4], linewidth=1.5)
                ax.axhline(thr, color=figures.OKABE_ITO[5], linewidth=2.0, linestyle="--",
                           label="threshold (99th pct healthy)")
                info = [r for r in (summ or [])
                        if r["test"] == test and r["condition"] == c
                        and r.get("featureset", fsname) == fsname]
                ttl = c.replace("_", " ") + chr(10) + fsname.replace("_", " ")
                if info:
                    al = fnum(info[0], "alarm_hours")
                    fa = fnum(info[0], "failure_hours")
                    if np.isfinite(fa):
                        ax.axvline(fa, color="black", linewidth=2.2, label="failure")
                    if np.isfinite(al):
                        ax.axvline(al, color=figures.OKABE_ITO[1], linewidth=2.2,
                                   linestyle="-.", label="first alarm")
                        if np.isfinite(fa):
                            ax.axvspan(al, fa, color=figures.OKABE_ITO[1], alpha=0.13)
                            ttl += "    lead = %.1f h" % (fa - al)
                    else:
                        ttl += "    NO ALARM"
                ax.set_title(ttl, fontsize=13)
                ax.set_yscale("log")
                if j == 0:
                    ax.set_ylabel("anomaly score")
        for j in range(len(fsets)):
            axes[-1][j].set_xlabel("Hours from start of run")
        axes[0][0].legend(loc="upper left", fontsize=10)
        fig.subplots_adjust(hspace=0.55, top=0.90)
        fig.suptitle("E8  Degradation trend, IMS %s (run to failure)" %
                     test.replace("_", " "), fontsize=18, y=0.995)
        figures.save(fig, os.path.join(FIGDIR, "fig_e8_trend_%s.png" % test))


def fig_scale_context():
    ppm = CFG["optics"]["pixels_per_meter"]
    amps_um = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500])
    px = amps_um * 1e-6 * ppm
    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.loglog(amps_um, px, color=figures.OKABE_ITO[4], marker="o", linewidth=2.6)
    for i, q in enumerate(CFG["camera"]["quantization_px"]):
        ax.axhline(q, color=figures.OKABE_ITO[(i % 3) + 1], linewidth=2.0,
                   linestyle=figures.LINESTYLES[(i % 4) + 1],
                   label="optical-flow step %.3f px" % q)
    ax.set_xlabel("Vibration displacement amplitude (um)")
    ax.set_ylabel("Image-plane amplitude (pixels)")
    ax.set_title("Scale check: 29 px/cm at 30 cm on 720p\n"
                 "a 50 um vibration spans only 0.145 pixels")
    ax.legend(loc="upper left", fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    figures.save(fig, os.path.join(FIGDIR, "fig_scale_context.png"))


def fig_e2b_speed_band():
    rows = load("e2b_speed_band.csv")
    if not rows:
        return
    rates = sorted(set(fnum(r, "fs_hz") for r in rows), reverse=True)
    full = [r for r in rows if fnum(r, "fs_hz") > 40000]
    ref = {r["band_lo"]: fnum(r, "roc_auc") for r in full}
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4))
    ax = axes[0]
    for i, fs in enumerate(rates):
        sel = sorted([r for r in rows if fnum(r, "fs_hz") == fs],
                     key=lambda r: fnum(r, "median_f0_hz"))
        if not sel:
            continue
        lab = ("full 48 kHz" if fs > 40000 else "%g Hz" % fs)
        ax.plot([fnum(r, "median_f0_hz") for r in sel],
                [fnum(r, "roc_auc") for r in sel], label=lab, **figures.style(i))
    figures.chance_line(ax)
    ax.set_xlabel("Shaft speed (Hz)")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.42, 1.02)
    ax.set_title("absolute performance" + chr(10) +
                 "low speed is hardest at EVERY rate, including full 48 kHz", fontsize=14)
    ax.legend(loc="lower right", fontsize=12)
    ax2 = axes[1]
    for i, fs in enumerate(rates):
        if fs > 40000:
            continue
        sel = sorted([r for r in rows if fnum(r, "fs_hz") == fs],
                     key=lambda r: fnum(r, "median_f0_hz"))
        xs, ys = [], []
        for r in sel:
            b = ref.get(r["band_lo"])
            if b is None:
                continue
            xs.append(fnum(r, "median_f0_hz"))
            ys.append(fnum(r, "roc_auc") - b)
        if xs:
            ax2.plot(xs, ys, label="%g Hz" % fs, **figures.style(i))
    ax2.axhline(0.0, color="black", linewidth=1.8)
    for rpm, txt in [(2400 / 60.0, "3x lost above 2400 rpm"),
                     (3600 / 60.0, "2x lost above 3600 rpm")]:
        ax2.axvline(rpm, color="#777777", linestyle=":", linewidth=2.0)
        ax2.text(rpm + 0.5, -0.29, txt, fontsize=11, rotation=90, va="bottom",
                 color="#444444")
    ax2.set_xlabel("Shaft speed (Hz)")
    ax2.set_ylabel("AUC change vs full 48 kHz")
    ax2.set_title("cost of the frame rate alone" + chr(10) +
                  "real, but not a simple function of speed", fontsize=14)
    ax2.legend(loc="lower left", fontsize=12)
    fig.suptitle("E2b  Performance by shaft speed - MAFAULDA, displacement",
                 fontsize=17, y=1.0)
    figures.save(fig, os.path.join(FIGDIR, "fig_e2b_speed_band.png"))


def fig_e2c_exposure():
    rows = load("e2c_exposure.csv")
    if not rows:
        return
    for d in sorted(set(r["dataset"] for r in rows)):
        sel = [r for r in rows if r["dataset"] == d]
        rates = sorted(set(fnum(r, "fs_hz") for r in sel), reverse=True)
        shutters = ["none", "1/1000 s", "1/500 s", "1/240 s"]
        present = [sh for sh in shutters if any(r["shutter"] == sh for r in sel)]
        fig, ax = plt.subplots(figsize=(10.5, 6.5))
        w = 0.8 / max(len(present), 1)
        xs = np.arange(len(rates))
        for i, sh in enumerate(present):
            v, e = [], []
            for fs in rates:
                m = [r for r in sel if fnum(r, "fs_hz") == fs and r["shutter"] == sh]
                v.append(fnum(m[0], "roc_auc_mean") if m else np.nan)
                e.append(fnum(m[0], "roc_auc_std", 0.0) if m else 0.0)
            lbl = "ideal anti-alias filter" if sh == "none" else "shutter " + sh
            ax.bar(xs + (i - (len(present) - 1) / 2.0) * w, v, w, yerr=e, capsize=4,
                   color=figures.OKABE_ITO[i + 1], edgecolor="black", linewidth=1.1,
                   hatch=["", "//", "..", "xx"][i % 4], label=lbl)
        figures.chance_line(ax)
        ax.set_xticks(xs)
        ax.set_xticklabels(["%g Hz" % r for r in rates])
        ax.set_ylabel("ROC-AUC")
        ax.set_ylim(0.4, 1.0)
        ax.set_title("E2c  Exposure time - " + d.upper() + chr(10) +
                     "a longer exposure is a stronger anti-alias filter")
        ax.legend(loc="upper right", fontsize=12)
        figures.save(fig, os.path.join(FIGDIR, "fig_e2c_exposure_%s.png" % d))


def fig_e2d_best_detector():
    rows = load("e2d_best_detector_overall.csv")
    if not rows:
        return
    for d in sorted(set(r["dataset"] for r in rows)):
        sel = [r for r in rows if r["dataset"] == d]
        fig, ax = plt.subplots(figsize=(10, 7))
        for i, det in enumerate(["ocsvm", "mahalanobis"]):
            pts = sorted([r for r in sel if r["detector"] == det
                          and fnum(r, "fs_hz") <= 40000],
                         key=lambda r: fnum(r, "fs_hz"))
            if not pts:
                continue
            full = [r for r in sel if r["detector"] == det and fnum(r, "fs_hz") > 40000]
            ref = fnum(full[0], "roc_auc_mean") if full else np.nan
            lab = ("OneClassSVM (best, E3)" if det == "ocsvm"
                   else "Mahalanobis (used elsewhere in this report)")
            ax.errorbar([fnum(r, "fs_hz") for r in pts],
                        [fnum(r, "roc_auc_mean") for r in pts],
                        yerr=[fnum(r, "roc_auc_std", 0.0) for r in pts],
                        capsize=4, label=lab, **figures.style(i + 1))
            if np.isfinite(ref):
                ax.axhline(ref, color=figures.OKABE_ITO[i + 1], linewidth=1.6,
                           linestyle=(0, (1, 1)), alpha=0.85)
                ax.text(min(RATES) * 1.04, ref + 0.012,
                        "full 48 kHz = %.3f" % ref, fontsize=12,
                        color=figures.OKABE_ITO[i + 1])
        figures.rate_axis(ax, RATES, primary=240)
        figures.chance_line(ax)
        ax.set_ylabel("ROC-AUC  (mean +/- s.d., 5-fold)")
        ax.set_ylim(0.45, 1.0)
        ax.set_title("E2d  The same sweep with the best detector - " + d.upper() +
                     chr(10) + "displacement, boxcar shutter; dotted = each model's own "
                     "full-rate reference")
        ax.legend(loc="lower right", fontsize=12.5, framealpha=0.95)
        figures.save(fig, os.path.join(FIGDIR, "fig_e2d_best_detector_%s.png" % d))


if __name__ == "__main__":
    for d in ["mafaulda", "cwru_48k"]:
        try:
            fig_e2_main(d)
            fig_e2_per_fault(d, "boxcar")
        except Exception as e:
            print("e2 skip", d, type(e).__name__, e)
    for fn in [fig_e1_domain, fig_e2b_speed_band, fig_e2c_exposure,
               fig_e2d_best_detector, fig_e3,
               fig_e4, fig_e5, fig_e6, fig_e7, fig_e8, fig_scale_context]:
        try:
            fn()
        except Exception as e:
            print("skip", fn.__name__, type(e).__name__, e)
    print("figures done")
