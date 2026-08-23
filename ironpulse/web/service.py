import io
import os
import sys
import threading
import time
import uuid

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
REPO = os.path.dirname(APP)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
if APP not in sys.path:
    sys.path.insert(0, APP)
os.environ.setdefault("IRONPULSE_PROJECT_ROOT", APP)

import cv2

from core import dsp
from appcore import video, markers, pipeline, cfgmap, calibrate
from . import baseline as bl
from . import db

CONFIG_PATH = os.path.join(APP, "config.yaml")


class ServiceError(Exception):
    pass


ENV_OVERRIDES = [
    ("PORT", ("web", "port"), int),
    ("HOST", ("web", "host"), str),
    ("IRONPULSE_OPEN_BROWSER", ("web", "open_browser"), lambda v: v.strip().lower()
     in ("1", "true", "yes", "on")),
    ("IRONPULSE_MAX_UPLOAD_MB", ("web", "max_upload_mb"), float),
    ("IRONPULSE_DATA_DIR", ("web", "data_dir"), str),
    ("IRONPULSE_DEMO_DIR", ("web", "demo_dir"), str),
    ("IRONPULSE_UPLOAD_DIR", ("web", "upload_dir"), str),
    ("IRONPULSE_DB_FILE", ("web", "db_file"), str),
]


def apply_env(cfg):
    applied = []
    for key, path, cast in ENV_OVERRIDES:
        raw = os.environ.get(key)
        if raw is None or raw == "":
            continue
        node = cfg
        for part in path[:-1]:
            node = node[part]
        try:
            node[path[-1]] = cast(raw)
        except (TypeError, ValueError):
            raise ServiceError(
                "ค่าของตัวแปรสภาพแวดล้อม %s = %r ใช้ไม่ได้ กับคีย์ %s"
                % (key, raw, ".".join(path)))
        applied.append("%s -> %s" % (key, ".".join(path)))
    return applied


def load_cfg():
    cfg = yaml.safe_load(io.open(CONFIG_PATH, encoding="utf-8"))
    apply_env(cfg)
    return cfg


def abspath(cfg, key):
    p = cfg["web"][key]
    return p if os.path.isabs(p) else os.path.join(APP, p)


def ensure_dirs(cfg):
    for k in ("data_dir", "demo_dir", "upload_dir"):
        d = abspath(cfg, k)
        if not os.path.isdir(d):
            os.makedirs(d)


def is_video(cfg, name):
    return os.path.splitext(name)[1].lower() in [
        s.lower() for s in cfg["web"]["video_suffixes"]]


def demo_files(cfg):
    d = abspath(cfg, "demo_dir")
    if not os.path.isdir(d):
        return []
    out = []
    for n in sorted(os.listdir(d)):
        if is_video(cfg, n):
            out.append(dict(name=n, path=os.path.join(d, n),
                            bytes=os.path.getsize(os.path.join(d, n))))
    return out


def session_of(created, prev_created, prev_id, gap_sec):
    import datetime as dt
    if not created:
        return prev_id + 1, None
    t = dt.datetime.strptime(created[:19], "%Y-%m-%dT%H:%M:%S")
    if prev_created is None or (t - prev_created).total_seconds() > gap_sec:
        return prev_id + 1, t
    return prev_id, t


def creation_time(path):
    import re
    import subprocess
    import imageio_ffmpeg
    r = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", path],
                       capture_output=True, text=True, errors="replace")
    m = re.search(r"creation_time\s*:\s*(\S+)", r.stderr or "")
    return m.group(1) if m else ""


def sessions_for(paths, gap_sec):
    stamped = []
    for p in paths:
        stamped.append((creation_time(p) or "", p))
    order = sorted(range(len(stamped)), key=lambda i: stamped[i][0])
    ids = [0] * len(stamped)
    prev_t, sid = None, 0
    for i in order:
        sid, prev_t = session_of(stamped[i][0], prev_t, sid, gap_sec)
        ids[i] = "S%d" % sid
    return ids


def first_frame_png(path, out_path):
    g = video.first_frame(path)
    cv2.imwrite(out_path, g)
    return g.shape[1], g.shape[0]


def analyse_one(cfg, path, roi_points, fs, blade_count, baseline_f0=None,
                progress=None):
    t0 = time.time()
    res = bl.analyse_clip(path, roi_points, cfg, fs, blade_count,
                          baseline_f0=baseline_f0, progress=progress)
    vec, names = bl.vector_of(res)
    axis = res["primary_axis"]
    grid, amp = bl.order_spectrum(res["signals"][axis], res["fs"], res["f0_hz"], cfg)
    pk = pipeline.peak_at_order(res["signals"][axis], res["fs"], cfg, res["f0_hz"])
    sig = res["signals"][axis]
    step = max(1, len(sig) // 1200)
    return dict(result=res, vector=vec, names=names, primary_axis=axis,
                f0_hz=float(res["f0_hz"]), fs=float(res["fs"]),
                blade_count=blade_count,
                spectrum=[float(v) for v in amp],
                spectrum_grid=[float(v) for v in grid],
                a1_px=float(pk["amp"]) if pk else float("nan"),
                a1_prominence=float(pk["prominence"]) if pk else float("nan"),
                wave=[float(v) for v in sig[::step]],
                wave_dt=float(step / res["fs"]),
                n_frames=int(res["n_frames"]),
                warnings=list(res["warnings"]),
                seconds=round(time.time() - t0, 2))


def unit_block(cfg, cal, a1_px):
    floor_px = float(cfg["scale"]["flow_resolution_px"])
    if cal is None:
        return dict(has_scale=False, a1_um=None, floor_um=None,
                    um_per_px=None, floor_px=floor_px)
    um = float(cal["um_per_px"])
    return dict(has_scale=True, a1_um=a1_px * um,
                floor_um=calibrate.detection_floor_um(cal, floor_px),
                um_per_px=um, floor_px=floor_px)


def scale_warnings(cfg, units, scale_id):
    out = []
    if not units["has_scale"]:
        out.append(
            "ยังไม่มีค่าสอบเทียบสเกลของการจัดเฟรมนี้ (scale_id %s) "
            "ระบบจึงแสดงผลเป็นพิกเซลล้วน ไม่แปลงเป็นไมโครเมตรให้ "
            "ให้ถ่ายคลิปที่มีไม้บรรทัดอยู่ในเฟรมเดียวกับการถ่ายจริง "
            "แล้วรัน verify/calibrate_scale.py" % (scale_id or "ไม่ได้ระบุ"))
        return out
    ratio = float(cfg["web"]["a1_floor_warn_ratio"])
    if units["a1_um"] is not None and units["floor_um"] > 0:
        if units["a1_um"] < ratio * units["floor_um"]:
            out.append(
                "แอมพลิจูดที่ 1x วัดได้ %.1f µm ต่ำกว่า %.1f เท่าของพื้นการวัด "
                "(%.1f µm) ผลครั้งนี้เชื่อถือได้น้อย "
                "ให้ขยับกล้องเข้าใกล้หรือซูมเข้าเพื่อลดไมโครเมตรต่อพิกเซล "
                "และเพิ่มแสงให้สว่างขึ้น"
                % (units["a1_um"], ratio, units["floor_um"]))
    return out


def inspect(cfg, con, machine, clip_path, filename, fs=None, progress=None):
    base = db.active_baseline(con, machine["id"])
    if base is None:
        raise ServiceError(
            "เครื่อง %s ยังไม่มี baseline ให้ไปหน้าสอบเทียบเครื่องใหม่ "
            "แล้วอัปโหลดคลิปสภาวะปกติอย่างน้อย %d คลิปก่อน"
            % (machine["name"], int(cfg["model"]["min_baseline_clips"])))
    fs = float(fs or machine["fs"])
    use_b = bool(cfg["analysis"]["use_baseline_f0"])
    a = analyse_one(cfg, clip_path, machine["roi"], fs, machine["blade_count"],
                    baseline_f0=float(base["f0_hz"]) if use_b else None,
                    progress=progress)
    det = bl.refit(cfg, base, machine["blade_count"])
    score = bl.score_vector(cfg, base, det, a["vector"], a["names"])
    thr = float(base["threshold"])
    cal = bl.scale_for(cfg, APP, machine["scale_id"])
    units = unit_block(cfg, cal, a["a1_px"])
    warnings = list(a["warnings"]) + scale_warnings(cfg, units, machine["scale_id"])
    ratio = score / thr if thr > 0 else float("inf")
    verdict = "ผิดปกติ" if score > thr else "ปกติ"
    payload = dict(
        machine_id=machine["id"], baseline_id=base["id"], filename=filename,
        fs=fs, f0_hz=a["f0_hz"], n_frames=a["n_frames"], score=score,
        threshold=thr, ratio=ratio, verdict=verdict, a1_px=a["a1_px"],
        a1_um=units["a1_um"], floor_um=units["floor_um"],
        primary_axis=a["primary_axis"], seconds=a["seconds"],
        detector_type=base["detector_type"], config_hash=base["config_hash"],
        core_version=base["core_version"], scale_id=machine["scale_id"],
        warnings=warnings,
        result=dict(spectrum=a["spectrum"], spectrum_grid=a["spectrum_grid"],
                    baseline_spectrum=base["spectrum"], wave=a["wave"],
                    wave_dt=a["wave_dt"], a1_prominence=a["a1_prominence"],
                    units=units, feature_set=base["feature_set"],
                    baseline_f0=float(base["f0_hz"]),
                    baseline_clips=int(base["n_clips"]),
                    fault_type="ความไม่สมดุลของมวลหมุน"))
    payload["id"] = db.add_inspection(con, payload)
    return payload


JOBS = {}
LOCK = threading.Lock()


def new_job(kind, total=1):
    jid = uuid.uuid4().hex[:12]
    with LOCK:
        JOBS[jid] = dict(id=jid, kind=kind, state="running", frames=0,
                         frames_total=0, step=0, step_total=total, message="",
                         result=None, error=None, started=time.time())
    return jid


def update(jid, **kw):
    with LOCK:
        if jid in JOBS:
            JOBS[jid].update(kw)


def job(jid):
    with LOCK:
        j = JOBS.get(jid)
        return dict(j) if j else None


def run_bg(jid, fn):
    def target():
        try:
            out = fn()
            update(jid, state="done", result=out,
                   seconds=round(time.time() - JOBS[jid]["started"], 2))
        except Exception as exc:
            update(jid, state="error", error=str(exc) or exc.__class__.__name__)
    t = threading.Thread(target=target, daemon=True)
    t.start()
    return t


def progress_cb(jid):
    def cb(n, total):
        update(jid, frames=int(n), frames_total=int(total))
    return cb
