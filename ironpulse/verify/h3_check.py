import os
import sys
import io
import csv
import glob
import time
import yaml
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
REPO = os.path.dirname(APP)
sys.path.insert(0, REPO)
sys.path.insert(0, APP)
os.environ.setdefault("IRONPULSE_PROJECT_ROOT", APP)

from appcore import video, markers, pipeline
import vibro_reference

CLIP_DIR = r"C:\Users\more_\Downloads\Arise H3"
CLICKS = [(330, 390), (330, 1393), (1007, 1754)]
TOL_HZ = 0.05
OUT = os.path.join(APP, "verify", "h3_results.csv")


def load_cfg():
    return yaml.safe_load(io.open(os.path.join(APP, "config.yaml"), encoding="utf-8"))


def main():
    cfg = load_cfg()
    fs = float(cfg["capture"]["default_fps"])
    margin = int(cfg["tracking"]["roi_margin_px"])
    clips = sorted(glob.glob(os.path.join(CLIP_DIR, "*.MOV")))
    rows = []
    print("%-14s %7s %9s %11s %11s %10s %8s %7s" %
          ("clip", "frames", "df_Hz", "app_peak", "ref_peak", "diff_Hz", "a1/noise", "sec"))
    for c in clips:
        name = os.path.basename(c)
        info = video.probe(c)
        rois = [markers.roi_from_point(x, y, margin, info["width"], info["height"])
                for x, y in CLICKS]
        g = video.first_frame(c)
        seeds = markers.seed_points(g, rois, cfg)
        t0 = time.time()
        res = pipeline.analyse(c, rois, cfg, fs=fs)
        el = time.time() - t0
        ref = vibro_reference.run(c, fs, [tuple(seeds[0]), tuple(seeds[2])])
        app_y = res["spectra"]["d_machine_y"]["peak_hz"]
        app_x = res["spectra"]["d_machine_x"]["peak_hz"]
        dy = abs(app_y - ref["y"]["peak_f"])
        dx = abs(app_x - ref["x"]["peak_f"])
        df = fs / res["n_frames"]
        rows.append(dict(clip=name, frames=res["n_frames"], freq_resolution_hz=round(df, 5),
                         app_peak_y_hz=round(app_y, 5), ref_peak_y_hz=round(ref["y"]["peak_f"], 5),
                         diff_y_hz=round(dy, 6),
                         app_peak_x_hz=round(app_x, 5), ref_peak_x_hz=round(ref["x"]["peak_f"], 5),
                         diff_x_hz=round(dx, 6),
                         pass_y=bool(dy <= TOL_HZ), pass_x=bool(dx <= TOL_HZ),
                         a1_over_noise=round(res["a1_over_noise"], 3),
                         f0_hz=round(res["f0_hz"], 4),
                         app_seconds=round(el, 2),
                         n_warnings=len(res["warnings"])))
        print("%-14s %7d %9.4f %11.4f %11.4f %10.6f %8.2f %7.1f" %
              (name, res["n_frames"], df, app_y, ref["y"]["peak_f"], dy,
               res["a1_over_noise"], el))
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    ny = sum(1 for r in rows if r["pass_y"])
    nx = sum(1 for r in rows if r["pass_x"])
    worst_y = max(r["diff_y_hz"] for r in rows)
    worst_x = max(r["diff_x_hz"] for r in rows)
    print()
    print("tolerance = %.3f Hz" % TOL_HZ)
    print("y axis (the axis 03_vibro.py uses): %d/%d clips pass, worst diff %.6f Hz"
          % (ny, len(rows), worst_y))
    print("x axis                            : %d/%d clips pass, worst diff %.6f Hz"
          % (nx, len(rows), worst_x))
    print("wrote", OUT)
    return 0 if (ny == len(rows) and nx == len(rows)) else 1


if __name__ == "__main__":
    sys.exit(main())
