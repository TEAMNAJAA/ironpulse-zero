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
CLIPS = ["IMG_8351", "IMG_8356", "IMG_8357"]
OUT = os.path.join(HERE, "crop_effect.csv")


def track_nocrop(path, rois, cfg, info):
    import cv2
    lk = dict(winSize=(int(cfg["tracking"]["win_size"]),) * 2,
              maxLevel=int(cfg["tracking"]["max_level"]),
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        int(cfg["tracking"]["max_iter"]),
                        float(cfg["tracking"]["epsilon"])),
              flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
              minEigThreshold=float(cfg["tracking"]["min_eig_threshold"]))
    prev, p0, out = None, None, []
    for g in video.frames(path, crop=None, width=info["width"], height=info["height"]):
        if prev is None:
            p0 = markers.seed_points(g, rois, cfg).reshape(-1, 1, 2)
            out.append(p0.reshape(-1, 2).copy())
            prev = g.copy()
            continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, p0, None, **lk)
        out.append(p1.reshape(-1, 2).copy())
        prev = g
        p0 = p1
    return np.array(out, dtype=np.float64)


def main():
    cfg = yaml.safe_load(io.open(os.path.join(APP, "config.yaml"), encoding="utf-8"))
    fs = float(cfg["capture"]["default_fps"])
    M = int(cfg["tracking"]["roi_margin_px"])
    rows = []
    print("%-14s %-3s %14s %14s %14s" %
          ("clip", "ax", "ref_peak", "app_crop", "app_nocrop"))
    for name in CLIPS:
        c = os.path.join(CLIP_DIR, name + ".MOV")
        info = video.probe(c)
        rois = [markers.roi_from_point(x, y, M, info["width"], info["height"])
                for x, y in CLICKS]
        g = video.first_frame(c)
        seeds = markers.seed_points(g, rois, cfg)
        t_crop, _ = markers.track(c, rois, cfg, info=info)
        t_full = track_nocrop(c, rois, cfg, info)
        ref = vibro_reference.run(c, fs, [tuple(seeds[0]), tuple(seeds[2])])
        d_crop = markers.differential(t_crop)
        d_full = markers.differential(t_full)
        for ax, key in [("y", "d_machine_y"), ("x", "d_machine_x")]:
            pk = {}
            for tag, d in [("crop", d_crop), ("nocrop", d_full)]:
                f, A = dsp.amplitude_spectrum(d[key], fs, mode=2, win="hann")
                m = f > 2
                pk[tag] = float(f[m][int(np.argmax(A[m]))])
            rows.append(dict(clip=name + ".MOV", axis=ax,
                             ref_peak_hz=round(ref[ax]["peak_f"], 4),
                             app_crop_hz=round(pk["crop"], 4),
                             app_nocrop_hz=round(pk["nocrop"], 4),
                             crop_matches=abs(pk["crop"] - ref[ax]["peak_f"]) <= 0.05,
                             nocrop_matches=abs(pk["nocrop"] - ref[ax]["peak_f"]) <= 0.05))
            print("%-14s %-3s %14.4f %14.4f %14.4f" %
                  (name, ax, ref[ax]["peak_f"], pk["crop"], pk["nocrop"]))
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    ys = [r for r in rows if r["axis"] == "y"]
    print()
    print("y axis: crop matches ref %d/%d ; nocrop matches ref %d/%d" %
          (sum(r["crop_matches"] for r in ys), len(ys),
           sum(r["nocrop_matches"] for r in ys), len(ys)))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
