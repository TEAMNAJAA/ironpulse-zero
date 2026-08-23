import os
import sys
import io
import csv
import yaml
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
REPO = os.path.dirname(APP)
sys.path.insert(0, REPO)
sys.path.insert(0, APP)
sys.path.insert(0, HERE)
os.environ.setdefault("IRONPULSE_PROJECT_ROOT", APP)

from appcore import video, markers
import vibro_reference
from core import dsp

CLIP_DIR = r"C:\Users\more_\Downloads\Arise H3"
CLICKS = [(330, 390), (330, 1393), (1007, 1754)]
CLIPS = ["IMG_8351", "IMG_8352", "IMG_8353", "IMG_8354",
         "IMG_8356", "IMG_8357", "IMG_8358", "IMG_8359"]
OUT = os.path.join(HERE, "spectrum_agreement.csv")


def main():
    cfg = yaml.safe_load(io.open(os.path.join(APP, "config.yaml"), encoding="utf-8"))
    fs = float(cfg["capture"]["default_fps"])
    M = int(cfg["tracking"]["roi_margin_px"])
    rows = []
    print("%-14s %-3s %11s %13s %9s %10s" %
          ("clip", "ax", "spec_corr", "max_rel_diff", "ref_rank", "tie_gap_%"))
    for name in CLIPS:
        c = os.path.join(CLIP_DIR, name + ".MOV")
        info = video.probe(c)
        rois = [markers.roi_from_point(x, y, M, info["width"], info["height"])
                for x, y in CLICKS]
        g = video.first_frame(c)
        seeds = markers.seed_points(g, rois, cfg)
        t, _ = markers.track(c, rois, cfg, info=info)
        d = markers.differential(t)
        ref = vibro_reference.run(c, fs, [tuple(seeds[0]), tuple(seeds[2])])
        for ax, key in [("y", "d_machine_y"), ("x", "d_machine_x")]:
            f, A = dsp.amplitude_spectrum(d[key], fs, mode=2, win="hann")
            rf, rA = ref[ax]["freq"], ref[ax]["spec"]
            n = min(len(A), len(rA))
            f2, A2, rA2 = f[:n], A[:n], rA[:n]
            m = f2 > 2
            corr = float(np.corrcoef(A2[m], rA2[m])[0, 1])
            rel = float(np.max(np.abs(A2[m] - rA2[m])) / np.max(rA2[m]))
            order = np.argsort(A2[m])[::-1]
            hit = np.where(np.isclose(f2[m][order], ref[ax]["peak_f"], atol=1e-6))[0]
            rank = int(hit[0]) + 1 if len(hit) else -1
            gap = (100.0 * (A2[m][order[0]] / A2[m][order[rank - 1]] - 1.0)
                   if rank > 0 else float("nan"))
            rows.append(dict(clip=name + ".MOV", axis=ax,
                             spectrum_corr=round(corr, 8),
                             max_rel_diff=float("%.3e" % rel),
                             ref_peak_rank_in_app=rank,
                             tie_gap_pct=round(gap, 3),
                             app_peak_hz=round(float(f2[m][order[0]]), 4),
                             ref_peak_hz=round(ref[ax]["peak_f"], 4)))
            print("%-14s %-3s %11.7f %13.2e %9d %10.3f" %
                  (name, ax, corr, rel, rank, gap))
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    ys = [r for r in rows if r["axis"] == "y"]
    xs = [r for r in rows if r["axis"] == "x"]
    print()
    print("y axis: min spectrum correlation %.7f, worst max_rel_diff %.2e" %
          (min(r["spectrum_corr"] for r in ys), max(r["max_rel_diff"] for r in ys)))
    print("x axis: min spectrum correlation %.7f, worst max_rel_diff %.2e" %
          (min(r["spectrum_corr"] for r in xs), max(r["max_rel_diff"] for r in xs)))
    print("worst tie gap on disagreeing y clips: %.3f %%" %
          max((r["tie_gap_pct"] for r in ys if r["ref_peak_rank_in_app"] > 1), default=0.0))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
