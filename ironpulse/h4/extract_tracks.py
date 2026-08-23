import os
import sys
import io
import csv
import time
import yaml
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
REPO = os.path.dirname(APP)
sys.path.insert(0, REPO)
sys.path.insert(0, APP)
os.environ.setdefault("IRONPULSE_PROJECT_ROOT", APP)

from core import dsp
from appcore import video, markers, pipeline

INV = os.path.join(HERE, "inventory.csv")
NPZ = os.path.join(HERE, "tracks.npz")
META = os.path.join(HERE, "tracks_meta.csv")
SHAFT_BAND = (10.0, 30.0)
PRIOR = [(530, 660, 170), (566, 1868, 140), (936, 1414, 170)]
CHECKER = 11
MIN_RESPONSE = 8000.0
TILE_COLS = (960, 1075)
SESSION_GAP_SEC = 120.0


def checker_kernel(sz):
    k = np.zeros((2 * sz, 2 * sz), np.float32)
    k[:sz, :sz] = 1.0
    k[sz:, sz:] = 1.0
    k[:sz, sz:] = -1.0
    k[sz:, :sz] = -1.0
    return k


def find_marker(gray, cx, cy, rad, ker):
    h, w = gray.shape
    y0, y1 = max(0, cy - rad), min(h, cy + rad)
    x0, x1 = max(0, cx - rad), min(w, cx + rad)
    sub = cv2.GaussianBlur(gray[y0:y1, x0:x1], (0, 0), 1.0)
    r = cv2.filter2D(sub, cv2.CV_32F, ker)
    i = np.unravel_index(int(np.argmax(np.abs(r))), r.shape)
    return int(x0 + i[1]), int(y0 + i[0]), float(abs(r[i]))


def tile_pitch(gray):
    pv = cv2.GaussianBlur(gray[:, TILE_COLS[0]:TILE_COLS[1]].mean(axis=1).reshape(1, -1),
                          (0, 0), 3).ravel()
    mins = []
    for i in range(15, 700):
        w = pv[i - 15:i + 16]
        if pv[i] == w.min() and (w.max() - pv[i]) > 4:
            mins.append(i)
    keep = []
    for i in mins:
        if not keep or i - keep[-1] > 40:
            keep.append(i)
    return float(keep[1] - keep[0]) if len(keep) >= 2 else float("nan")


def sessions(rows):
    import datetime as dt
    order = sorted(rows, key=lambda r: r["created"])
    sid, out, prev = 0, {}, None
    for r in order:
        t = dt.datetime.strptime(r["created"], "%Y-%m-%dT%H:%M:%S.%f%z")
        if prev is None or (t - prev).total_seconds() > SESSION_GAP_SEC:
            sid += 1
        out[r["clip"]] = "S%d" % sid
        prev = t
    return out


def main():
    cfg = yaml.safe_load(io.open(os.path.join(APP, "config.yaml"), encoding="utf-8"))
    fs = float(cfg["capture"]["default_fps"])
    M = int(cfg["tracking"]["roi_margin_px"])
    order = int(cfg["analysis"]["detrend_order"])
    blades = cfg["machine"]["blade_count"]
    ker = checker_kernel(CHECKER)
    rows = [r for r in csv.DictReader(io.open(INV, encoding="utf-8"))
            if r["group"] != "calibration"]
    sess = sessions(rows)
    rows.sort(key=lambda r: (r["group"], r["clip"]))
    tracks, meta = {}, []
    for r in rows:
        path = r["path"]
        info = video.probe(path)
        g = video.first_frame(path).astype(np.float32)
        pts, resp = [], []
        for cx, cy, rad in PRIOR:
            x, y, v = find_marker(g, cx, cy, rad, ker)
            pts.append((x, y))
            resp.append(v)
        rois = [markers.roi_from_point(x, y, M, info["width"], info["height"])
                for x, y in pts]
        t0 = time.time()
        res = pipeline.analyse(path, rois, cfg, f0_range=SHAFT_BAND, blade_count=blades)
        el = time.time() - t0
        t = res["track"]
        tracks[r["clip"]] = t.astype(np.float64)
        mach, base, ref = t[:, 0, :], t[:, 1, :], t[:, 2, :]
        row = dict(clip=r["clip"], group=r["group"], severity=r["severity"],
                   session=sess[r["clip"]], created=r["created"],
                   n_frames_total=info["n_frames"], n_tracked=int(t.shape[0]),
                   tracking_lost=bool(t.shape[0] < video.frames_used(info, cfg)),
                   f0_hz=round(res["f0_hz"], 4), f0_source=res["f0_source"],
                   primary_axis=res["primary_axis"],
                   a1_over_noise=round(res["a1_over_noise"], 3),
                   n_warnings=len(res["warnings"]),
                   seed_machine_x=pts[0][0], seed_machine_y=pts[0][1],
                   seed_base_x=pts[1][0], seed_base_y=pts[1][1],
                   seed_ref_x=pts[2][0], seed_ref_y=pts[2][1],
                   resp_machine=round(resp[0], 1), resp_base=round(resp[1], 1),
                   resp_ref=round(resp[2], 1),
                   weak_marker=bool(min(resp) < MIN_RESPONSE),
                   sep_machine_base=round(float(np.hypot(*(t[0, 0] - t[0, 1]))), 3),
                   sep_machine_ref=round(float(np.hypot(*(t[0, 0] - t[0, 2]))), 3),
                   tile_pitch_px=round(tile_pitch(g), 2),
                   mach_rms=round(float(np.hypot(*[dsp.poly_detrend(mach[:, k], order).std()
                                                   for k in (0, 1)])), 5),
                   base_rms=round(float(np.hypot(*[dsp.poly_detrend(base[:, k], order).std()
                                                   for k in (0, 1)])), 5),
                   ref_rms=round(float(np.hypot(*[dsp.poly_detrend(ref[:, k], order).std()
                                                  for k in (0, 1)])), 5),
                   seconds=round(el, 2))
        meta.append(row)
        print("%-10s %-8s %-4s f0=%7.3f  prim=%s  a1/n=%8.1f  tile=%6.1f  %.1fs" %
              (row["clip"], row["group"], row["session"], row["f0_hz"],
               row["primary_axis"].split("_")[-1], row["a1_over_noise"],
               row["tile_pitch_px"], el), flush=True)
    np.savez_compressed(NPZ, **tracks)
    with open(META, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(meta[0].keys()))
        w.writeheader()
        w.writerows(meta)
    print()
    print("wrote", NPZ)
    print("wrote", META, len(meta), "clips")


if __name__ == "__main__":
    main()
