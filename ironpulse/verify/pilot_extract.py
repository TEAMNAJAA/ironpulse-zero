import os
import sys
import io
import csv
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

CLIP_DIR = r"C:\Users\more_\Downloads\Arise H3"
CLICKS = [(330, 390), (330, 1393), (1007, 1754)]
CLIPS = ["IMG_8351", "IMG_8352", "IMG_8353", "IMG_8354",
         "IMG_8356", "IMG_8357", "IMG_8358", "IMG_8359"]
SHAFT_BAND = (10.0, 30.0)
NPZ = os.path.join(HERE, "pilot_data.npz")
CSV = os.path.join(HERE, "pilot_raw.csv")


def main():
    cfg = yaml.safe_load(io.open(os.path.join(APP, "config.yaml"), encoding="utf-8"))
    fs = float(cfg["capture"]["default_fps"])
    M = int(cfg["tracking"]["roi_margin_px"])
    rows, vectors, spectra, names_ref = [], [], {}, None
    for name in CLIPS:
        c = os.path.join(CLIP_DIR, name + ".MOV")
        info = video.probe(c)
        rois = [markers.roi_from_point(x, y, M, info["width"], info["height"])
                for x, y in CLICKS]
        t0 = time.time()
        res = pipeline.analyse(c, rois, cfg, f0_range=SHAFT_BAND)
        el = time.time() - t0
        t = res["track"]
        vec, names = res["feature_vector"]
        names_ref = names
        vectors.append(vec)
        sx = res["spectra"]["d_machine_x"]
        spectra[name] = np.vstack([sx["freq"] / res["f0_hz"], sx["amp"]])
        free = pipeline.auto_f0(res["signals"], fs, cfg, None)
        px = pipeline.peak_of(res["signals"]["d_machine_x"], fs, cfg)
        py = pipeline.peak_of(res["signals"]["d_machine_y"], fs, cfg)
        row = dict(clip=name,
                   n_frames_total=info["n_frames"],
                   n_frames_used=video.frames_used(info, cfg),
                   n_tracked=int(t.shape[0]),
                   tracking_lost=bool(t.shape[0] < video.frames_used(info, cfg)),
                   freq_resolution_hz=round(fs / t.shape[0], 5),
                   f0_shaftband_hz=round(res["f0_hz"], 4),
                   f0_shaftband_rpm=round(res["f0_hz"] * 60, 1),
                   f0_source=res["f0_source"],
                   f0_unconstrained_hz=round(free[0], 4),
                   f0_unconstrained_axis=free[2] if isinstance(free[2], str) else "",
                   peak_x_hz=round(px["freq"], 4),
                   peak_x_prom=round(px["prominence"], 4),
                   peak_y_hz=round(py["freq"], 4),
                   peak_y_prom=round(py["prominence"], 4),
                   machine_p2p_x=round(float(np.ptp(t[:, 0, 0])), 5),
                   machine_p2p_y=round(float(np.ptp(t[:, 0, 1])), 5),
                   base_p2p_x=round(float(np.ptp(t[:, 1, 0])), 5),
                   base_p2p_y=round(float(np.ptp(t[:, 1, 1])), 5),
                   ref_p2p_x=round(float(np.ptp(t[:, 2, 0])), 5),
                   ref_p2p_y=round(float(np.ptp(t[:, 2, 1])), 5),
                   diff_machine_p2p_x=round(float(np.ptp(res["signals"]["d_machine_x"])), 5),
                   diff_machine_p2p_y=round(float(np.ptp(res["signals"]["d_machine_y"])), 5),
                   diff_base_p2p_y=round(float(np.ptp(res["signals"]["d_base_y"])), 5),
                   a1_over_noise=round(res["a1_over_noise"], 3),
                   n_warnings=len(res["warnings"]),
                   seconds=round(el, 2))
        for k in ("r1", "r2", "r3", "r05", "E_sub", "E_1x", "E_mid", "E_hi",
                  "harm", "centroid", "crest", "kurt", "rbp"):
            row["x." + k] = round(float(res["features"]["d_machine_x"][k]), 6)
            row["y." + k] = round(float(res["features"]["d_machine_y"][k]), 6)
            row["base_y." + k] = round(float(res["features"]["d_base_y"][k]), 6)
        rows.append(row)
        print("%-12s frames %d->%d tracked %d  f0=%7.3f Hz (%s)  x_peak=%7.3f prom=%.2f  "
              "r1x=%.4f  %.1fs" %
              (name, info["n_frames"], row["n_frames_used"], row["n_tracked"],
               res["f0_hz"], res["f0_source"], px["freq"], px["prominence"],
               row["x.r1"], el), flush=True)
    with open(CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(NPZ, vectors=np.vstack(vectors),
                        names=np.array(names_ref, dtype=object),
                        clips=np.array(CLIPS, dtype=object),
                        **{"spec_" + k: v for k, v in spectra.items()})
    print()
    print("wrote", CSV)
    print("wrote", NPZ)


if __name__ == "__main__":
    main()
