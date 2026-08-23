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

NORMAL = ["IMG_8351", "IMG_8352", "IMG_8353", "IMG_8354", "IMG_8356"]
CLAY = ["IMG_8357", "IMG_8358", "IMG_8359"]
NPZ = os.path.join(HERE, "pilot_data.npz")
CSV = os.path.join(HERE, "pilot_raw.csv")
OUT_D = os.path.join(HERE, "pilot_cohens_d.csv")
FIG = os.path.join(HERE, "pilot_spectra.png")


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp == 0:
        return np.inf if a.mean() != b.mean() else 0.0
    return (a.mean() - b.mean()) / sp


def main():
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    by = {r["clip"]: r for r in rows}
    d = np.load(NPZ, allow_pickle=True)
    clips = list(d["clips"])
    names = list(d["names"])
    V = d["vectors"]
    gi = {c: i for i, c in enumerate(clips)}
    ni = [gi[c] for c in NORMAL]
    ci = [gi[c] for c in CLAY]

    print("=" * 96)
    print("1. r1 on the signal axis (x)")
    print("=" * 96)
    print("%-12s %-8s %10s %10s %12s %10s" %
          ("clip", "group", "r1_x", "r1_y", "f0_Hz", "a1/noise"))
    for c in clips:
        r = by[c]
        g = "normal" if c in NORMAL else "clay"
        print("%-12s %-8s %10.4f %10.4f %12.3f %10s" %
              (c, g, float(r["x.r1"]), float(r["y.r1"]),
               float(r["f0_shaftband_hz"]), r["a1_over_noise"]))
    rn = np.array([float(by[c]["x.r1"]) for c in NORMAL])
    rc = np.array([float(by[c]["x.r1"]) for c in CLAY])
    print()
    print("  normal mean r1_x = %.4f  (sd %.4f, n=%d)" % (rn.mean(), rn.std(ddof=1), len(rn)))
    print("  clay   mean r1_x = %.4f  (sd %.4f, n=%d)" % (rc.mean(), rc.std(ddof=1), len(rc)))
    print("  ratio clay/normal = %.3f x" % (rc.mean() / rn.mean()))
    print("  SPEC 6.9 rule fires at r1_ratio >= 2.0  ->  %s" %
          ("FIRES" if rc.mean() / rn.mean() >= 2.0 else "does NOT fire"))

    print()
    print("=" * 96)
    print("2. Cohen's d, all features, sorted by |d|")
    print("=" * 96)
    out = []
    for j, nm in enumerate(names):
        dd = cohens_d(V[ci, j], V[ni, j])
        out.append((abs(dd) if np.isfinite(dd) else -1, dd, nm,
                    float(V[ni, j].mean()), float(V[ci, j].mean())))
    out.sort(reverse=True)
    print("%-22s %10s %12s %12s" % ("feature", "cohen_d", "normal_mean", "clay_mean"))
    for a, dd, nm, mn, mc in out[:15]:
        print("%-22s %10.3f %12.5f %12.5f" % (nm, dd, mn, mc))
    with open(OUT_D, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["feature", "cohens_d", "abs_d", "normal_mean", "clay_mean"])
        for a, dd, nm, mn, mc in out:
            w.writerow([nm, round(dd, 5) if np.isfinite(dd) else "", round(a, 5),
                        round(mn, 6), round(mc, 6)])
    best = out[0]
    print()
    print("  best separator: %s  (d = %.3f)" % (best[2], best[1]))

    print()
    print("=" * 96)
    print("3. peak-to-peak, machine marker vs reference marker (raw tracked pixels)")
    print("=" * 96)
    print("%-12s %-8s %9s %9s %9s %9s %10s" %
          ("clip", "group", "mach_x", "mach_y", "ref_x", "ref_y", "mach/ref"))
    for c in clips:
        r = by[c]
        g = "normal" if c in NORMAL else "clay"
        mx, my = float(r["machine_p2p_x"]), float(r["machine_p2p_y"])
        rx, ry = float(r["ref_p2p_x"]), float(r["ref_p2p_y"])
        ratio = np.hypot(mx, my) / max(np.hypot(rx, ry), 1e-9)
        print("%-12s %-8s %9.4f %9.4f %9.4f %9.4f %10.2f" %
              (c, g, mx, my, rx, ry, ratio))

    print()
    print("=" * 96)
    print("4. frames tracked")
    print("=" * 96)
    print("%-12s %-8s %9s %9s %9s %8s %12s" %
          ("clip", "group", "total", "used", "tracked", "lost", "df_Hz"))
    for c in clips:
        r = by[c]
        g = "normal" if c in NORMAL else "clay"
        print("%-12s %-8s %9s %9s %9s %8s %12s" %
              (c, g, r["n_frames_total"], r["n_frames_used"], r["n_tracked"],
               r["tracking_lost"], r["freq_resolution_hz"]))
    lost = [c for c in clips if by[c]["tracking_lost"] == "True"]
    print()
    print("  clips with tracking dropout: %s" % (", ".join(lost) if lost else "none"))
    dfs = sorted(set(by[c]["freq_resolution_hz"] for c in clips))
    print("  distinct frequency resolutions across the 8 clips: %s" % dfs)

    figures.use()
    fig, ax = plt.subplots(figsize=(12.5, 7))
    for i, c in enumerate(clips):
        s = d["spec_" + c]
        order, amp = s[0], s[1]
        m = (order >= 0.15) & (order <= 6.0)
        st = figures.style(i)
        ax.semilogy(order[m], np.maximum(amp[m], 1e-7), color=st["color"],
                    linestyle=st["linestyle"], linewidth=1.6, alpha=0.9,
                    label="%s  r1=%.3f" % (c, float(by[c]["x.r1"])))
    for k in (1, 2, 3):
        ax.axvline(k, color="#bbbbbb", linewidth=1.2, zorder=0)
        ax.text(k, ax.get_ylim()[1] * 0.5, "%dx" % k, ha="center", fontsize=13,
                color="#777777")
    ax.set_xlabel("Order (multiples of shaft speed)")
    ax.set_ylabel("amplitude (px)")
    ax.set_title("Pilot: order spectrum of d_machine_x, 8 clips at 2000 frames each"
                 + chr(10) + "one colour per clip - group labels not yet supplied")
    ax.set_xlim(0.15, 6.0)
    ax.legend(loc="upper right", fontsize=11, ncol=2)
    figures.save(fig, FIG)
    print()
    print("wrote", OUT_D)


if __name__ == "__main__":
    main()
