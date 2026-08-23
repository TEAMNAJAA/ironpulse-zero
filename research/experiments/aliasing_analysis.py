import os, sys, csv
import numpy as np
import matplotlib.pyplot as plt
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)
sys.path.insert(0, os.path.dirname(_PROJ))
from experiments.common import CFG, write_csv, table
from core import figures

figures.use()
FIGDIR = CFG["paths"]["figures"]
FPS = 240.0
NYQ = FPS / 2.0

LINES = [
    ("CWRU", "FTF cage", 0.39828),
    ("CWRU", "BPFO outer race", 3.5848),
    ("CWRU", "BSF ball", 4.7135),
    ("CWRU", "BPFI inner race", 5.4152),
    ("MAFAULDA", "FTF cage", 0.3750),
    ("MAFAULDA", "BSF ball", 1.8710),
    ("MAFAULDA", "BPFO outer race", 2.9980),
    ("MAFAULDA", "BPFI inner race", 5.0020),
]
SHAFT_HZ = [12.0, 20.0, 24.0, 29.9, 40.0, 50.0, 58.5]
EXPOSURES = [("1/240 s", 1 / 240.0), ("1/500 s", 1 / 500.0), ("1/1000 s", 1 / 1000.0)]
CONFUSABLE = {1.0: "1x (imbalance)", 2.0: "2x (misalignment)", 3.0: "3x (misalignment)"}


def fold(order, nyq_order):
    k = round(order / (2.0 * nyq_order))
    return abs(order - 2.0 * nyq_order * k)


def sinc_gain(f_hz, exposure_s):
    x = np.pi * f_hz * exposure_s
    return 1.0 if x == 0 else abs(np.sin(x) / x)


def nearest_confusable(a):
    best, bd = None, 1e9
    for o, n in CONFUSABLE.items():
        d = abs(a - o)
        if d < bd:
            best, bd = n, d
    return best, bd


rows = []
for ds, name, order in LINES:
    for f0 in SHAFT_HZ:
        f_hz = order * f0
        nyq_order = NYQ / f0
        kept = order <= nyq_order
        a = order if kept else fold(order, nyq_order)
        conf, dist = nearest_confusable(a)
        r = dict(dataset=ds, line=name, order=order, shaft_hz=f0,
                 shaft_rpm=round(f0 * 60), true_hz=round(f_hz, 2),
                 nyquist_order=round(nyq_order, 3),
                 status="kept" if kept else "aliased",
                 apparent_order=round(a, 4),
                 nearest_confusable=("" if kept else conf),
                 distance_to_confusable=("" if kept else round(dist, 3)))
        for lab, e in EXPOSURES:
            r["shutter_gain_" + lab.replace("/", "_").replace(" ", "")] = round(
                sinc_gain(f_hz, e), 4)
        rows.append(r)

write_csv(table("t", "aliasing_analysis.csv"), rows)

danger = [r for r in rows if r["status"] == "aliased" and r["distance_to_confusable"] != ""
          and float(r["distance_to_confusable"]) < 0.35]
print("Aliased lines landing within 0.35 order of a real machine signature:")
for r in sorted(danger, key=lambda r: float(r["distance_to_confusable"]))[:18]:
    print("  %-9s %-16s %4.0f rpm: order %.3f -> %.3f  (%s, gap %.3f)  1/240s shutter keeps %.0f%%"
          % (r["dataset"], r["line"], r["shaft_rpm"], r["order"], r["apparent_order"],
             r["nearest_confusable"], float(r["distance_to_confusable"]),
             100 * r["shutter_gain_1_240s"]))

fig, ax = plt.subplots(figsize=(10.5, 6.5))
f = np.linspace(1, 1200, 4000)
for i, (lab, e) in enumerate(EXPOSURES):
    g = np.array([sinc_gain(x, e) for x in f])
    ax.plot(f, g, label="exposure %s" % lab, **dict(figures.style(i + 1), markevery=400))
ax.axvline(NYQ, color="black", linewidth=2.2, linestyle="-", alpha=0.7)
ax.text(NYQ * 1.05, 0.93, "Nyquist\n120 Hz", fontsize=13)
ax.axvspan(NYQ, f[-1], color="#bbbbbb", alpha=0.3, zorder=0)
ax.set_xscale("log")
ax.set_xlabel("Vibration frequency (Hz)")
ax.set_ylabel("Shutter (boxcar) gain")
ax.set_ylim(0, 1.05)
ax.set_title("Why a SLOW shutter protects you at 240 fps\n"
             "grey = frequencies that fold back and fake a fault; "
             "gain there is the alias amplitude")
ax.legend(loc="lower left", fontsize=13)
figures.save(fig, os.path.join(FIGDIR, "fig_shutter_antialias.png"))

fig, ax = plt.subplots(figsize=(11, 7))
for i, (ds, name, order) in enumerate(LINES):
    if ds != "MAFAULDA":
        continue
    xs = np.linspace(10, 62, 300)
    ap = [o if o <= NYQ / x else fold(o, NYQ / x) for x, o in zip(xs, [order] * len(xs))]
    ax.plot(xs, ap, label="%s (%.3fx)" % (name, order),
            markevery=25, **figures.style(i))
for o, n in CONFUSABLE.items():
    ax.axhline(o, color="#555555", linewidth=1.6, linestyle=(0, (4, 3)))
    ax.text(10.4, o + 0.06, n, fontsize=12, color="#333333")
ax.set_xlabel("Shaft speed (Hz)")
ax.set_ylabel("Order the camera actually reports")
ax.set_title("MAFAULDA bearing lines at 240 fps without anti-aliasing\n"
             "where each defect frequency APPEARS to be")
ax.set_ylim(0, 5.6)
ax.legend(loc="upper right", fontsize=12)
figures.save(fig, os.path.join(FIGDIR, "fig_alias_masquerade.png"))
print("aliasing analysis done")
