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

from core import dsp
from appcore import video, markers, pipeline

CLIP_DIR = r"C:\Users\more_\Downloads\Arise H3 fix"
CLIPS = ["normal_01", "normal_02", "normal_03",
         "unbal_01", "unbal_02", "unbal_03"]
CLICKS = {
    "normal_01": [(523, 929), (844, 1643)],
    "normal_02": [(523, 929), (845, 1643)],
    "normal_03": [(523, 929), (844, 1643)],
    "unbal_01": [(502, 956), (900, 1697)],
    "unbal_02": [(502, 954), (898, 1696)],
    "unbal_03": [(502, 952), (899, 1696)],
}
SHAFT_BAND = (10.0, 30.0)
NPZ = os.path.join(HERE, "pilot2_data.npz")
CSV = os.path.join(HERE, "pilot2_raw.csv")


def pearson(a, b, order):
    a = dsp.poly_detrend(np.asarray(a, float), order)
    b = dsp.poly_detrend(np.asarray(b, float), order)
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0:
        return float("nan")
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def main():
    cfg = yaml.safe_load(io.open(os.path.join(APP, "config.yaml"), encoding="utf-8"))
    fs = float(cfg["capture"]["default_fps"])
    M = int(cfg["tracking"]["roi_margin_px"])
    order = int(cfg["analysis"]["detrend_order"])
    blades = cfg["machine"]["blade_count"]
    rows, vectors, spectra, names_ref = [], [], {}, None
    for name in CLIPS:
        c = os.path.join(CLIP_DIR, name + ".MOV")
        info = video.probe(c)
        rois = [markers.roi_from_point(x, y, M, info["width"], info["height"])
                for x, y in CLICKS[name]]
        t0 = time.time()
        res = pipeline.analyse(c, rois, cfg, f0_range=SHAFT_BAND,
                               blade_count=blades)
        el = time.time() - t0
        t = res["track"]
        vec, names = res["feature_vector"]
        names_ref = names
        vectors.append(vec)
        for ax in ("x", "y"):
            s = res["spectra"]["d_machine_" + ax]
            spectra["%s|%s" % (name, ax)] = np.vstack([s["freq"] / res["f0_hz"], s["amp"]])
        free = pipeline.auto_f0(res["signals"], fs, cfg, None)
        px = pipeline.peak_of(res["signals"]["d_machine_x"], fs, cfg)
        py = pipeline.peak_of(res["signals"]["d_machine_y"], fs, cfg)
        mx, my = t[:, 0, 0], t[:, 0, 1]
        rx, ry = t[:, 1, 0], t[:, 1, 1]
        row = dict(clip=name,
                   group="normal" if name.startswith("normal") else "unbal",
                   n_frames_total=info["n_frames"],
                   n_frames_used=video.frames_used(info, cfg),
                   n_tracked=int(t.shape[0]),
                   tracking_lost=bool(t.shape[0] < video.frames_used(info, cfg)),
                   freq_resolution_hz=round(fs / t.shape[0], 5),
                   f0_shaftband_hz=round(res["f0_hz"], 4),
                   f0_shaftband_rpm=round(res["f0_hz"] * 60, 1),
                   f0_source=res["f0_source"],
                   primary_axis=res["primary_axis"],
                   blade_count=res["blade_count"],
                   rbp_blade_count=res["rbp_blade_count"],
                   rbp_available=bool(res["available"]["d_machine_x"]["rbp"]),
                   f0_unconstrained_hz=round(free[0], 4),
                   f0_unconstrained_axis=free[1],
                   f0_unconstrained_prom=round(free[2], 3),
                   peak_x_hz=round(px["freq"], 4),
                   peak_x_amp=round(px["amp"], 6),
                   peak_x_prom=round(px["prominence"], 4),
                   peak_y_hz=round(py["freq"], 4),
                   peak_y_amp=round(py["amp"], 6),
                   peak_y_prom=round(py["prominence"], 4),
                   machine_p2p_x=round(float(np.ptp(mx)), 5),
                   machine_p2p_y=round(float(np.ptp(my)), 5),
                   ref_p2p_x=round(float(np.ptp(rx)), 5),
                   ref_p2p_y=round(float(np.ptp(ry)), 5),
                   machine_rms_x=round(float(dsp.poly_detrend(mx, order).std()), 5),
                   machine_rms_y=round(float(dsp.poly_detrend(my, order).std()), 5),
                   ref_rms_x=round(float(dsp.poly_detrend(rx, order).std()), 5),
                   ref_rms_y=round(float(dsp.poly_detrend(ry, order).std()), 5),
                   corr_x=round(pearson(mx, rx, order), 4),
                   corr_y=round(pearson(my, ry, order), 4),
                   diff_machine_p2p_x=round(float(np.ptp(res["signals"]["d_machine_x"])), 5),
                   diff_machine_p2p_y=round(float(np.ptp(res["signals"]["d_machine_y"])), 5),
                   a1_over_noise=round(res["a1_over_noise"], 3),
                   seed_machine_x=round(float(t[0, 0, 0]), 2),
                   seed_machine_y=round(float(t[0, 0, 1]), 2),
                   seed_ref_x=round(float(t[0, 1, 0]), 2),
                   seed_ref_y=round(float(t[0, 1, 1]), 2),
                   n_warnings=len(res["warnings"]),
                   seconds=round(el, 2))
        for k in ("r1", "r2", "r3", "r05", "E_sub", "E_1x", "E_mid", "E_hi",
                  "harm", "centroid", "crest", "kurt", "rbp"):
            for sig, pre in (("d_machine_x", "x."), ("d_machine_y", "y.")):
                row[pre + k] = round(float(res["features"][sig][k]), 6)
        rows.append(row)
        print("%-11s frames %d->%d tracked %d  f0=%7.3f Hz (%s)  "
              "x_peak=%7.3f prom=%5.2f  y_peak=%7.3f prom=%5.2f  "
              "r1x=%.4f r1y=%.4f  %.1fs" %
              (name, info["n_frames"], row["n_frames_used"], row["n_tracked"],
               res["f0_hz"], res["f0_source"], px["freq"], px["prominence"],
               py["freq"], py["prominence"], row["x.r1"], row["y.r1"], el),
              flush=True)
        for w in res["warnings"]:
            print("      ! " + w, flush=True)
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
