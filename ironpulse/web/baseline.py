import hashlib
import io
import os

import numpy as np

from core import dsp, detectors
from appcore import cfgmap, markers, pipeline, calibrate

CORE_FILES = ["dsp.py", "features.py", "detectors.py", "config.py", "signals.py",
              "camera_sim.py", "evaluate.py", "figures.py", "ims.py", "pipeline.py"]


class BaselineError(Exception):
    pass


def config_hash(app_cfg_path):
    return hashlib.md5(io.open(app_cfg_path, "rb").read()).hexdigest()[:12]


def core_version(repo_root):
    h = hashlib.md5()
    for name in CORE_FILES:
        p = os.path.join(repo_root, "core", name)
        if os.path.isfile(p):
            h.update(io.open(p, "rb").read().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:12]


def shaft_band(app_cfg):
    lo, hi = app_cfg["analysis"]["shaft_band_hz"]
    return float(lo), float(hi)


def detector_cfg(app_cfg, fs, f0, blade_count):
    return cfgmap.core_cfg(app_cfg, fs, f0, blade_count)


def feature_columns(names, feature_set, primary_axis):
    if feature_set == "all":
        return list(range(len(names)))
    if feature_set == "machine":
        return [i for i, n in enumerate(names) if n.startswith("d_machine_")]
    if feature_set == "primary":
        return [i for i, n in enumerate(names) if n.startswith(primary_axis + ".")]
    raise BaselineError("ไม่รู้จักชุด feature ชื่อ %s" % feature_set)


def fit_detector(app_cfg, X, fs, f0, blade_count, seed):
    name = app_cfg["model"]["type"]
    if name not in detectors.BY_NAME:
        raise BaselineError("config model.type = %s ไม่มีในระบบตรวจจับที่รู้จัก" % name)
    ccfg = detector_cfg(app_cfg, fs, f0, blade_count)
    return detectors.BY_NAME[name](ccfg, seed=seed).fit(np.asarray(X, dtype=np.float64))


def threshold_of(app_cfg, det):
    t = float(det.threshold()) * float(app_cfg["model"]["threshold_margin"])
    if not np.isfinite(t) or t <= 0:
        t = float(np.max(det.train_scores))
    return t


def order_spectrum(sig, fs, f0, app_cfg):
    f, A = dsp.amplitude_spectrum(sig, fs,
                                  mode=int(app_cfg["analysis"]["detrend_order"]),
                                  win=app_cfg["analysis"]["window_fn"])
    order = f / float(f0)
    hi = float(app_cfg["analysis"]["spectrum_order_max"])
    n = int(app_cfg["analysis"]["spectrum_points"])
    grid = np.linspace(0.15, hi, n)
    m = (order >= 0.1) & (order <= hi * 1.05)
    return grid, np.interp(grid, order[m], A[m])


def analyse_clip(path, roi_points, app_cfg, fs, blade_count, baseline_f0=None,
                 progress=None):
    from appcore import video
    info = video.probe(path)
    margin = int(app_cfg["tracking"]["roi_margin_px"])
    gray = video.first_frame(path)
    points, weak = markers.reacquire_all(gray, roi_points, app_cfg)
    rois = [markers.roi_from_point(x, y, margin, info["width"], info["height"])
            for x, y in points]
    lo, hi = shaft_band(app_cfg)
    res = pipeline.analyse(path, rois, app_cfg, fs=fs, blade_count=blade_count,
                           baseline_f0=baseline_f0, progress=progress,
                           f0_range=(lo, hi))
    res["seed_points"] = [[float(a), float(b)] for a, b in points]
    res["seed_shift_px"] = [float(np.hypot(q[0] - p[0], q[1] - p[1]))
                            for p, q in zip(roi_points, points)]
    if weak:
        res["warnings"].append(
            "หามาร์กเกอร์ลายตารางไม่เจอที่จุด %s ใกล้ตำแหน่งที่บันทึกไว้ "
            "ระบบจึงใช้ตำแหน่งเดิมตามที่คลิกไว้ "
            "ถ้าย้ายกล้องหรือเปลี่ยนแสงไปมาก ให้กดตั้งมาร์กเกอร์ใหม่ก่อนตรวจ"
            % ", ".join(str(w[0]) for w in weak))
    big = [i + 1 for i, d in enumerate(res["seed_shift_px"])
           if d > float(app_cfg["tracking"]["reacquire_radius_px"]) * 0.6]
    if big:
        res["warnings"].append(
            "มาร์กเกอร์จุด %s ขยับจากตำแหน่งที่บันทึกไว้มาก (%s px) "
            "แปลว่ากล้องหรือเครื่องถูกย้ายไปจากตอนสอบเทียบ "
            "สเกลอาจเปลี่ยนตามไปด้วย ให้สอบเทียบใหม่ก่อนเชื่อค่าเป็นไมโครเมตร"
            % (", ".join(str(i) for i in big),
               ", ".join("%.0f" % res["seed_shift_px"][i - 1] for i in big)))
    return res


def vector_of(res):
    vec, names = res["feature_vector"]
    return np.asarray(vec, dtype=np.float64), list(names)


def primary_axis_vote(results):
    counts = {}
    for r in results:
        counts[r["primary_axis"]] = counts.get(r["primary_axis"], 0) + 1
    return max(sorted(counts), key=lambda k: counts[k])


def build(results, app_cfg, scale_id, config_hash_value, core_version_value,
          sessions):
    if not results:
        raise BaselineError("ไม่มีคลิปสำหรับสร้าง baseline")
    need = int(app_cfg["model"]["min_baseline_clips"])
    warnings = []
    if len(results) < need:
        raise BaselineError(
            "มีคลิปปกติ %d คลิป ต้องการอย่างน้อย %d คลิป ขาดอีก %d คลิป "
            "จึงจะสร้าง baseline ได้" % (len(results), need, need - len(results)))
    warn_at = int(app_cfg["model"]["warn_baseline_clips"])
    if len(results) < warn_at:
        warnings.append(
            "มีคลิปปกติ %d คลิป ต่ำกว่าจำนวนที่แนะนำ %d คลิป "
            "ค่าเกณฑ์จะแกว่งกว่าที่ควร ควรถ่ายเพิ่ม" % (len(results), warn_at))
    uniq = sorted(set(sessions))
    if len(uniq) < 2:
        warnings.append(
            "คลิปปกติทั้ง %d คลิปมาจากรอบการถ่ายเดียวกัน "
            "H4 พิสูจน์แล้วว่า baseline ที่กระจุกในรอบเดียวทำให้เกณฑ์เพี้ยน "
            "ควรถ่ายเพิ่มอย่างน้อย 2 รอบ โดยตั้งกล้องใหม่ในแต่ละรอบ" % len(results))
    else:
        big = max(sessions.count(s) for s in uniq)
        if big > 0.6 * len(results):
            warnings.append(
                "คลิปปกติ %d จาก %d คลิปมาจากรอบการถ่ายเดียว "
                "ควรกระจายให้ใกล้เคียงกันทุกรอบ อย่างน้อย 5 รอบ รอบละประมาณ %d คลิป"
                % (big, len(results), max(1, len(results) // 5)))

    names = list(results[0]["names"])
    axis = primary_axis_vote(results)
    cols = feature_columns(names, app_cfg["model"]["feature_set"], axis)
    V = np.vstack([r["vector"] for r in results])
    X = V[:, cols]
    f0 = float(np.median([r["f0_hz"] for r in results]))
    fs = float(results[0]["fs"])
    seed = int(app_cfg["model"]["random_state"])
    det = fit_detector(app_cfg, X, fs, f0, results[0].get("blade_count"), seed)
    thr = threshold_of(app_cfg, det)
    spec = np.vstack([r["spectrum"] for r in results])
    f0s = np.array([r["f0_hz"] for r in results], dtype=np.float64)
    spread = float(f0s.max() - f0s.min()) / f0 if f0 > 0 else 0.0
    if spread > float(app_cfg["analysis"]["f0_shift_warn"]):
        warnings.append(
            "ความเร็วรอบของคลิป baseline ต่างกัน %.1f%% (%.2f ถึง %.2f Hz) "
            "เกินเกณฑ์ %.0f%% ควรคุมความเร็วรอบให้คงที่ตอนถ่าย baseline"
            % (100 * spread, f0s.min(), f0s.max(),
               100 * float(app_cfg["analysis"]["f0_shift_warn"])))
    return dict(fs=fs, f0_hz=f0, n_clips=len(results),
                detector_type=app_cfg["model"]["type"],
                feature_set=app_cfg["model"]["feature_set"],
                feature_names=[names[i] for i in cols],
                train_vectors=[[float(v) for v in row] for row in X],
                feature_mean=[float(v) for v in X.mean(axis=0)],
                feature_std=[float(v) for v in X.std(axis=0, ddof=1)]
                if len(X) > 1 else [0.0] * X.shape[1],
                threshold=thr, seed=seed,
                spectrum=[float(v) for v in np.median(spec, axis=0)],
                sessions=list(sessions), config_hash=config_hash_value,
                core_version=core_version_value, scale_id=scale_id,
                warnings=warnings, primary_axis=axis)


def refit(app_cfg, baseline, blade_count):
    X = np.asarray(baseline["train_vectors"], dtype=np.float64)
    return fit_detector(app_cfg, X, float(baseline["fs"]), float(baseline["f0_hz"]),
                        blade_count, int(baseline["seed"]))


def score_vector(app_cfg, baseline, det, vector, names):
    want = list(baseline["feature_names"])
    index = {n: i for i, n in enumerate(names)}
    missing = [n for n in want if n not in index]
    if missing:
        raise BaselineError(
            "คลิปนี้ให้ feature ไม่ครบตามที่ baseline ใช้ ขาด %s "
            "มักเกิดจากจำนวนมาร์กเกอร์ไม่เท่ากับตอนสร้าง baseline"
            % ", ".join(missing))
    x = np.array([[vector[index[n]] for n in want]], dtype=np.float64)
    return float(det.score(x)[0])


def scale_for(app_cfg, app_dir, scale_id):
    cals = calibrate.load(os.path.join(app_dir, app_cfg["scale"]["calibration_file"]))
    if not scale_id:
        return None
    return cals.get(str(scale_id))
