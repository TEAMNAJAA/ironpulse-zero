import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
REPO = os.path.dirname(APP)
sys.path.insert(0, REPO)
sys.path.insert(0, APP)
os.environ.setdefault("IRONPULSE_PROJECT_ROOT", APP)

from core import figures

NORMAL = ["normal_01", "normal_02", "normal_03"]
UNBAL = ["unbal_01", "unbal_02", "unbal_03"]
PREV_CV_PCT = 16.8
F0_TOL = 0.05
NPZ = os.path.join(HERE, "pilot2_data.npz")
CSV = os.path.join(HERE, "pilot2_raw.csv")
OUT_D = os.path.join(HERE, "pilot2_cohens_d.csv")
FIG = os.path.join(HERE, "pilot2_spectra.png")
LINE = "=" * 104


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp == 0:
        return np.inf if a.mean() != b.mean() else 0.0
    return (a.mean() - b.mean()) / sp


def cv(a):
    a = np.asarray(a, float)
    return 100.0 * a.std(ddof=1) / a.mean()


def main():
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    by = {r["clip"]: r for r in rows}
    d = np.load(NPZ, allow_pickle=True)
    clips = list(d["clips"])
    names = list(d["names"])
    V = d["vectors"]
    gi = {c: i for i, c in enumerate(clips)}
    ni = [gi[c] for c in NORMAL]
    ui = [gi[c] for c in UNBAL]

    def g(c):
        return "normal" if c in NORMAL else "unbal"

    def fl(c, k):
        return float(by[c][k])

    print(LINE)
    print("0. run sanity")
    print(LINE)
    print("%-11s %-8s %8s %8s %9s %7s %9s %10s" %
          ("clip", "group", "frames", "used", "tracked", "lost", "df_Hz", "a1/noise"))
    for c in clips:
        r = by[c]
        print("%-11s %-8s %8s %8s %9s %7s %9s %10s" %
              (c, g(c), r["n_frames_total"], r["n_frames_used"], r["n_tracked"],
               r["tracking_lost"], r["freq_resolution_hz"], r["a1_over_noise"]))
    lost = [c for c in clips if by[c]["tracking_lost"] == "True"]
    print()
    print("  tracking dropout : %s" % (", ".join(lost) if lost else "none"))
    print("  df across clips  : %s" % sorted(set(by[c]["freq_resolution_hz"] for c in clips)))
    print("  primary axis     : %s" %
          ", ".join("%s=%s" % (c, by[c]["primary_axis"].split("_")[-1]) for c in clips))
    print("  blade_count      : %s (measured, blade pass 57.0 Hz / f0 18.96 Hz = 3.006)"
          % by[clips[0]]["blade_count"])
    print("  markers in setup : 2 (machine on motor housing, reference on wall)")
    print("  d_base absent    : feature vector is %d values, not 39" % len(names))

    print()
    print(LINE)
    print("1. r1 per clip, group means, between-group ratio")
    print(LINE)
    print("%-11s %-8s %10s %10s %12s %12s %10s %10s" %
          ("clip", "group", "r1_x", "r1_y", "A1x_px", "A1y_px", "prom_x", "prom_y"))
    for c in clips:
        print("%-11s %-8s %10.4f %10.4f %12.5f %12.5f %10.2f %10.2f" %
              (c, g(c), fl(c, "x.r1"), fl(c, "y.r1"), fl(c, "peak_x_amp"),
               fl(c, "peak_y_amp"), fl(c, "peak_x_prom"), fl(c, "peak_y_prom")))
    print()
    for ax in ("x", "y"):
        rn = np.array([fl(c, ax + ".r1") for c in NORMAL])
        ru = np.array([fl(c, ax + ".r1") for c in UNBAL])
        print("  axis %s : normal %.4f (sd %.4f, n=%d) | unbal %.4f (sd %.4f, n=%d) "
              "| ratio %.3f x" %
              (ax, rn.mean(), rn.std(ddof=1), len(rn),
               ru.mean(), ru.std(ddof=1), len(ru), ru.mean() / rn.mean()))
    print()
    for ax in ("x", "y"):
        an = np.array([fl(c, "peak_%s_amp" % ax) for c in NORMAL])
        au = np.array([fl(c, "peak_%s_amp" % ax) for c in UNBAL])
        print("  axis %s : 1x amplitude normal %.5f px | unbal %.5f px | ratio %.1f x" %
              (ax, an.mean(), au.mean(), au.mean() / an.mean()))
    print()
    print("  r1 = A1 / sqrt(sum(A^2)/2), so a pure 1x sinusoid gives r1 = sqrt(2) = 1.4142")
    for ax in ("x", "y"):
        rn = np.array([fl(c, ax + ".r1") for c in NORMAL])
        print("  axis %s : ceiling on any achievable r1 ratio = 1.4142 / %.4f = %.3f x" %
              (ax, rn.mean(), 1.4142136 / rn.mean()))
    print("  the retired SPEC 6.9 rule fired at r1_ratio >= 2.0")
    for ax in ("x", "y"):
        rn = np.array([fl(c, ax + ".r1") for c in NORMAL])
        ru = np.array([fl(c, ax + ".r1") for c in UNBAL])
        print("    axis %s observed %.3f x -> %s" %
              (ax, ru.mean() / rn.mean(),
               "would fire" if ru.mean() / rn.mean() >= 2.0 else "would NOT fire"))
    print("  replaced by rules.z_threshold, see verify/rules_check.py")

    print()
    print(LINE)
    print("2. within-group spread of the normal group, against the previous pilot")
    print(LINE)
    print("%-14s %10s %10s %10s %6s" % ("quantity", "mean", "sd", "cv_pct", "n"))
    for ax in ("x", "y"):
        a = np.array([fl(c, ax + ".r1") for c in NORMAL])
        print("%-14s %10.4f %10.4f %10.1f %6d" %
              ("r1_" + ax, a.mean(), a.std(ddof=1), cv(a), len(a)))
    for ax in ("x", "y"):
        a = np.array([fl(c, "peak_%s_amp" % ax) for c in NORMAL])
        print("%-14s %10.5f %10.5f %10.1f %6d" %
              ("A1_" + ax + "_px", a.mean(), a.std(ddof=1), cv(a), len(a)))
    print()
    for ax in ("x", "y"):
        a = np.array([fl(c, ax + ".r1") for c in UNBAL])
        print("  unbal group r1_%s cv = %.1f %%" % (ax, cv(a)))
    print()
    cvx = cv(np.array([fl(c, "x.r1") for c in NORMAL]))
    print("  previous pilot (8 clips, old setup, r1_x) cv = %.1f %%" % PREV_CV_PCT)
    print("  this pilot     (3 clips, new setup, r1_x) cv = %.1f %%" % cvx)
    print("  change = %+.1f percentage points, %.2f x the old spread" %
          (cvx - PREV_CV_PCT, cvx / PREV_CV_PCT))
    print("  n = 3, so this sd carries 2 degrees of freedom, read it as an observation")

    print()
    print(LINE)
    print("3. peak to peak and correlation, machine marker against reference marker, raw px")
    print(LINE)
    print("%-11s %-8s %8s %8s %8s %8s %9s %9s %9s %9s" %
          ("clip", "group", "mach_x", "mach_y", "ref_x", "ref_y",
           "p2p_ratio", "rms_ratio", "corr_x", "corr_y"))
    for c in clips:
        mx, my = fl(c, "machine_p2p_x"), fl(c, "machine_p2p_y")
        rx, ry = fl(c, "ref_p2p_x"), fl(c, "ref_p2p_y")
        mr = np.hypot(fl(c, "machine_rms_x"), fl(c, "machine_rms_y"))
        rr = np.hypot(fl(c, "ref_rms_x"), fl(c, "ref_rms_y"))
        print("%-11s %-8s %8.4f %8.4f %8.4f %8.4f %9.2f %9.2f %9.4f %9.4f" %
              (c, g(c), mx, my, rx, ry,
               np.hypot(mx, my) / max(np.hypot(rx, ry), 1e-9), mr / max(rr, 1e-9),
               fl(c, "corr_x"), fl(c, "corr_y")))
    print()
    for grp, cl in (("normal", NORMAL), ("unbal", UNBAL)):
        pr = [np.hypot(fl(c, "machine_p2p_x"), fl(c, "machine_p2p_y")) /
              np.hypot(fl(c, "ref_p2p_x"), fl(c, "ref_p2p_y")) for c in cl]
        print("  %-6s mean p2p ratio %6.2f x  (min %.2f, max %.2f)" %
              (grp, float(np.mean(pr)), min(pr), max(pr)))
    cors = [fl(c, "corr_" + a) for c in clips for a in ("x", "y")]
    print("  correlation machine vs reference over all clips and both axes: %.3f to %.3f" %
          (min(cors), max(cors)))
    bad = [c for c in clips
           if np.hypot(fl(c, "ref_p2p_x"), fl(c, "ref_p2p_y")) >
           np.hypot(fl(c, "machine_p2p_x"), fl(c, "machine_p2p_y"))]
    print("  clips where the reference marker moves MORE than the machine marker: %s" %
          (", ".join(bad) if bad else "none"))
    print("  markers.reference_check now runs at 2 points and fires on the clips above")

    print()
    print(LINE)
    print("4. Cohen d, all %d features, sorted by absolute value, unbal minus normal" % len(names))
    print(LINE)
    out = []
    for j, nm in enumerate(names):
        dd = cohens_d(V[ui, j], V[ni, j])
        out.append((abs(dd) if np.isfinite(dd) else 1e18, dd, nm,
                    float(V[ni, j].mean()), float(V[ui, j].mean())))
    out.sort(reverse=True)
    print("%-24s %12s %14s %14s" % ("feature", "cohen_d", "normal_mean", "unbal_mean"))
    for a, dd, nm, mn, mu in out:
        print("%-24s %12s %14.5f %14.5f" %
              (nm, ("%.3f" % dd) if np.isfinite(dd) else "inf", mn, mu))
    with open(OUT_D, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["feature", "cohens_d", "abs_d", "normal_mean", "unbal_mean"])
        for a, dd, nm, mn, mu in out:
            w.writerow([nm, round(dd, 5) if np.isfinite(dd) else "inf",
                        round(a, 5) if a < 1e17 else "inf", round(mn, 6), round(mu, 6)])
    fin = [o for o in out if np.isfinite(o[1])]
    print()
    if fin:
        print("  best finite separator: %s (d = %.3f)" % (fin[0][2], fin[0][1]))
    print("  abs d >= 0.8 : %d of %d features" % (sum(1 for o in out if o[0] >= 0.8), len(out)))
    print("  n = 3 per group, d is descriptive here, no inference attached")
    gated = by[clips[0]]["rbp_blade_count"] in ("", "None")
    blades = int(by[clips[0]]["blade_count"])
    print("  rbp gated off: %s (blade_count = %d)" % (gated, blades))
    if gated:
        print("  reason: the blade pass order lands exactly on shaft harmonic %d, which is"
              % blades)
        print("          feature r%d already, so rbp duplicated r%d value for value and could"
              % (blades, blades))
        print("          not separate blade damage from misalignment")
        print("  rbp reads 0.0 in both groups and means NOT MEASURED, not no change")

    print()
    print(LINE)
    print("5. shaft speed per clip, tolerance %.0f %%" % (100 * F0_TOL))
    print(LINE)
    print("%-11s %-8s %12s %10s %14s %11s %20s" %
          ("clip", "group", "f0_Hz", "rpm", "f0_free_Hz", "free_axis", "source"))
    for c in clips:
        print("%-11s %-8s %12.4f %10.1f %14.4f %11s %20s" %
              (c, g(c), fl(c, "f0_shaftband_hz"), fl(c, "f0_shaftband_rpm"),
               fl(c, "f0_unconstrained_hz"), by[c]["f0_unconstrained_axis"],
               by[c]["f0_source"]))
    f0 = np.array([fl(c, "f0_shaftband_hz") for c in clips])
    df = float(by[clips[0]]["freq_resolution_hz"])
    spread = (f0.max() - f0.min()) / f0.mean()
    print()
    print("  min %.4f Hz  max %.4f Hz  mean %.4f Hz  sd %.4f Hz  spread %.2f %%" %
          (f0.min(), f0.max(), f0.mean(), f0.std(ddof=1), 100 * spread))
    print("  fft bin width df = %.4f Hz, so min and max sit %.1f bin apart" %
          (df, (f0.max() - f0.min()) / df))
    worst, wc = 0.0, None
    for c in clips:
        s = abs(fl(c, "f0_shaftband_hz") - f0.mean()) / f0.mean()
        if s > worst:
            worst, wc = s, c
    print("  largest deviation from the mean: %s at %.2f %%" % (wc, 100 * worst))
    if spread > F0_TOL:
        print("  ALERT: spread exceeds %.0f %%, features are not comparable across clips" %
              (100 * F0_TOL))
    else:
        print("  OK: spread is inside %.0f %%, no f0 alert" % (100 * F0_TOL))
    off = [c for c in clips
           if abs(fl(c, "f0_unconstrained_hz") - fl(c, "f0_shaftband_hz")) > df]
    print("  clips where the unconstrained peak is not the shaft peak: %s" %
          (", ".join(off) if off else "none"))

    figures.use()
    fig, axes = plt.subplots(2, 1, figsize=(13, 12), sharex=True)
    col = {"normal": "#0072B2", "unbal": "#D55E00"}
    ls = ["-", "--", "-."]
    for k, axis_name in enumerate(("x", "y")):
        ax = axes[k]
        for i, c in enumerate(clips):
            s = d["spec_%s|%s" % (c, axis_name)]
            order, amp = s[0], s[1]
            m = (order >= 0.15) & (order <= 6.0)
            ax.semilogy(order[m], np.maximum(amp[m], 1e-7),
                        color=col[g(c)], linestyle=ls[i % 3], linewidth=1.7,
                        alpha=0.9,
                        label="%s  r1=%.3f" % (c, fl(c, "%s.r1" % axis_name)))
        for kk in (1, 2, 3):
            ax.axvline(kk, color="#bbbbbb", linewidth=1.2, zorder=0)
        ax.set_ylabel("amplitude (px)")
        ax.set_title("d_machine_%s" % axis_name)
        ax.set_xlim(0.15, 6.0)
        ax.legend(loc="upper right", fontsize=11, ncol=2)
    axes[1].set_xlabel("Order (multiples of shaft speed)")
    fig.suptitle("Pilot 2 order spectra, 6 clips at 2000 frames"
                 + chr(10) + "blue = normal, orange = unbalanced", y=0.945)
    figures.save(fig, FIG)
    print()
    print("wrote", OUT_D)


if __name__ == "__main__":
    main()
