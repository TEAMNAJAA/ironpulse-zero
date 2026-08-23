import os
import sys
import io
import csv
import yaml
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
REPO = os.path.dirname(APP)
sys.path.insert(0, REPO)
sys.path.insert(0, APP)
os.environ.setdefault("IRONPULSE_PROJECT_ROOT", APP)

from appcore import video, calibrate

CLIP = r"C:\Users\more_\Downloads\Arise H4\calibration_ruler\IMG_8379.MOV"
SETUP_ID = "h4_wall_2026_08_22"
SEARCH = dict(y0=1630, y1=1730, x0=310, x1=800)
TICK_OFFSETS = (17, 28)
EDGE_MIN_GRAD = 20.0
PERIOD_RANGE = (15.0, 45.0)
PERIOD_STEPS = 6001
N_LAGS = 11
CM_PER_TICK = 1.0
OUT = os.path.join(HERE, "calibration.csv")


def fit_edge(sub):
    b = cv2.GaussianBlur(sub, (0, 0), 1.0)
    gy = cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)
    xs, ys = [], []
    for x in range(sub.shape[1]):
        col = gy[:, x]
        i = int(np.argmin(col))
        if -col[i] > EDGE_MIN_GRAD:
            xs.append(float(x))
            ys.append(float(i))
    xs, ys = np.array(xs), np.array(ys)
    p = np.polyfit(xs, ys, 1)
    for _ in range(8):
        r = ys - np.polyval(p, xs)
        s = 1.4826 * np.median(np.abs(r - np.median(r)))
        keep = np.abs(r - np.median(r)) < max(3 * s, 1.5)
        p = np.polyfit(xs[keep], ys[keep], 1)
    r = ys - np.polyval(p, xs)
    keep = np.abs(r) < 2.0
    return p, int(keep.sum()), len(xs), float(r[keep].std())


def strip(sub, p, lo, hi):
    rows = []
    for off in range(lo, hi):
        v = []
        for x in range(sub.shape[1]):
            y = np.polyval(p, x) + off
            y0 = int(np.floor(y))
            fr = y - y0
            v.append((1 - fr) * sub[y0, x] + fr * sub[y0 + 1, x]
                     if 0 <= y0 < sub.shape[0] - 1 else np.nan)
        rows.append(v)
    return np.array(rows, dtype=np.float64).mean(axis=0)


def local_norm(v, w=61):
    k = np.ones(w) / w
    m = np.convolve(v, k, mode="same")
    s = np.sqrt(np.maximum(np.convolve((v - m) ** 2, k, mode="same"), 1e-9))
    z = (v - m) / s
    z[:w // 2] = 0.0
    z[-(w // 2):] = 0.0
    return z


def dominant_period(z):
    t = np.arange(len(z), dtype=np.float64)
    pers = np.linspace(PERIOD_RANGE[0], PERIOD_RANGE[1], PERIOD_STEPS)
    amp = np.array([np.hypot((z * np.cos(2 * np.pi * t / P)).sum(),
                             (z * np.sin(2 * np.pi * t / P)).sum()) for P in pers])
    i = int(np.argmax(amp))
    far = np.abs(pers - pers[i]) > 2.0
    prom = float(amp[i] / amp[far].max()) if far.any() else float("inf")
    return float(pers[i]), prom


def autocorr_pitch(z, seed):
    n = len(z)
    ac = np.correlate(z, z, mode="full")[n - 1:]
    ac = ac / ac[0]
    ks, lags, wts = [], [], []
    for k in range(1, N_LAGS + 1):
        c = int(round(k * seed))
        lo, hi = max(1, c - 6), min(len(ac), c + 7)
        if hi <= lo:
            continue
        j = lo + int(np.argmax(ac[lo:hi]))
        ks.append(float(k))
        lags.append(float(j))
        wts.append(max(float(ac[j]), 0.0))
    p = np.polyfit(np.array(ks), np.array(lags), 1, w=np.array(wts))
    return float(p[0]), list(zip(ks, lags, wts))


def main():
    cfg = yaml.safe_load(io.open(os.path.join(APP, "config.yaml"), encoding="utf-8"))
    cal_path = os.path.join(APP, cfg["scale"]["calibration_file"])
    flow_px = float(cfg["scale"]["flow_resolution_px"])
    info = video.probe(CLIP)
    g = video.first_frame(CLIP).astype(np.float32)
    sub = g[SEARCH["y0"]:SEARCH["y1"], SEARCH["x0"]:SEARCH["x1"]]
    p, inliers, total, rms = fit_edge(sub)
    angle = float(np.degrees(np.arctan(p[0])))
    print("ruler edge : angle %.3f deg, %d/%d inliers, rms %.2f px"
          % (angle, inliers, total, rms))

    v = strip(sub, p, TICK_OFFSETS[0], TICK_OFFSETS[1])
    z = local_norm(v)
    seed, prom = dominant_period(z)
    pitch_x, lags = autocorr_pitch(z, seed)
    n = len(z)
    left, _ = dominant_period(local_norm(v[:n // 2]))
    right, _ = dominant_period(local_norm(v[n // 2:]))
    print("periodicity: seed %.3f px (prominence %.2f), autocorr slope %.4f px"
          % (seed, prom, pitch_x))
    print("halves     : left %.3f px, right %.3f px, spread %.2f %%"
          % (left, right, 100 * abs(left - right) / pitch_x))
    print("lag table  : " + "  ".join("k%d=%d" % (int(k), int(l)) for k, l, _ in lags))

    cosang = np.cos(np.radians(angle))
    pitch_axis = pitch_x / cosang
    span_cm = float(N_LAGS)
    dx = pitch_x * span_cm
    dy = p[0] * dx
    y_ref = np.polyval(p, 0.0) + 0.5 * (TICK_OFFSETS[0] + TICK_OFFSETS[1])
    p1 = (SEARCH["x0"] + 0.0, SEARCH["y0"] + y_ref)
    p2 = (SEARCH["x0"] + dx, SEARCH["y0"] + y_ref + dy)
    pairs = [(p1, p2, span_cm * 10.0 * CM_PER_TICK)]
    cal = calibrate.from_clip(CLIP, pairs, SETUP_ID,
                             "steel ruler, %d cm span read from %d autocorrelation lags"
                             % (int(span_cm), N_LAGS))
    cal["ruler_angle_deg"] = angle
    cal["pitch_px_per_cm_along_axis"] = pitch_axis
    cal["half_left_px_per_cm"] = left / cosang
    cal["half_right_px_per_cm"] = right / cosang
    cal["perspective_spread_frac"] = abs(left - right) / pitch_x
    calibrate.store(cal, cal_path)

    print()
    print("px per mm  : %.4f" % cal["px_per_mm"])
    print("um per px  : %.2f" % cal["um_per_px"])
    print("floor      : flow %.3f px = %.2f um" %
          (flow_px, calibrate.detection_floor_um(cal, flow_px)))
    print("perspective: %.2f %% between the two halves of the ruler"
          % (100 * cal["perspective_spread_frac"]))
    print("wrote      : %s" % cal_path)

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "value", "unit"])
        w.writerow(["ruler_angle", round(angle, 4), "deg"])
        w.writerow(["edge_fit_rms", round(rms, 4), "px"])
        w.writerow(["tick_pitch_column", round(pitch_x, 4), "px"])
        w.writerow(["tick_pitch_axis", round(pitch_axis, 4), "px_per_cm"])
        w.writerow(["px_per_mm", round(cal["px_per_mm"], 5), "px/mm"])
        w.writerow(["um_per_px", round(cal["um_per_px"], 3), "um/px"])
        w.writerow(["half_left", round(cal["half_left_px_per_cm"], 4), "px_per_cm"])
        w.writerow(["half_right", round(cal["half_right_px_per_cm"], 4), "px_per_cm"])
        w.writerow(["perspective_spread", round(100 * cal["perspective_spread_frac"], 3), "pct"])
        w.writerow(["detection_floor", round(calibrate.detection_floor_um(cal, flow_px), 3), "um"])
        w.writerow(["width", info["width"], "px"])
        w.writerow(["height", info["height"], "px"])
    print("wrote      : %s" % OUT)


if __name__ == "__main__":
    main()
