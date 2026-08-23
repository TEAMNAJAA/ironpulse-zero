import io
import json
import os
import time

import numpy as np

from . import video


class CalibrationError(Exception):
    pass


def px_distance(p1, p2):
    return float(np.hypot(float(p2[0]) - float(p1[0]), float(p2[1]) - float(p1[1])))


def px_per_mm(p1, p2, known_mm):
    known_mm = float(known_mm)
    if known_mm <= 0:
        raise CalibrationError("ระยะจริงบนไม้บรรทัดต้องมากกว่า 0 มิลลิเมตร")
    d = px_distance(p1, p2)
    if d <= 0:
        raise CalibrationError("จุดสองจุดบนไม้บรรทัดซ้อนกันอยู่ ต้องคลิกให้ห่างกัน")
    return d / known_mm


def measure(pairs, app_cfg=None):
    vals = [px_per_mm(p1, p2, mm) for p1, p2, mm in pairs]
    a = np.asarray(vals, dtype=np.float64)
    spread = float(a.max() - a.min()) / float(a.mean()) if len(a) > 1 else 0.0
    return dict(px_per_mm=float(a.mean()), n_pairs=len(a),
                px_per_mm_min=float(a.min()), px_per_mm_max=float(a.max()),
                spread_frac=spread, values=[float(v) for v in a])


def from_clip(path, pairs, setup_id, notes=""):
    info = video.probe(path)
    m = measure(pairs)
    m.update(setup_id=str(setup_id), width=info["width"], height=info["height"],
             source=os.path.basename(path), notes=str(notes),
             um_per_px=1000.0 / m["px_per_mm"],
             created_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    return m


def load(path):
    if not os.path.isfile(path):
        return {}
    return json.loads(io.open(path, encoding="utf-8").read())


def save(cals, path):
    io.open(path, "w", encoding="utf-8", newline="\n").write(
        json.dumps(cals, ensure_ascii=False, indent=2, sort_keys=True))


def store(cal, path):
    cals = load(path)
    cals[cal["setup_id"]] = cal
    save(cals, path)
    return cals


def require(cals, setup_id, info=None):
    if setup_id is None:
        raise CalibrationError(
            "ยังไม่ได้ระบุ setup_id ของการจัดเฟรม จึงแปลงพิกเซลเป็นไมโครเมตรไม่ได้ "
            "ให้ถ่ายคลิปสอบเทียบที่มีไม้บรรทัดอยู่ในเฟรมเดียวกับการถ่ายจริง "
            "แล้วรัน verify/calibrate_scale.py")
    cal = cals.get(str(setup_id))
    if cal is None:
        raise CalibrationError(
            "ไม่มีค่าสอบเทียบสเกลของ setup_id %s "
            "ค่า 42.9 px/cm ในเอกสารเดิมวัดจากการจัดเฟรมชุดก่อน ใช้กับชุดนี้ไม่ได้ "
            "ให้ถ่ายคลิปสอบเทียบที่มีไม้บรรทัดแล้วรัน verify/calibrate_scale.py"
            % setup_id)
    if info is not None:
        if int(cal["width"]) != int(info["width"]) or int(cal["height"]) != int(info["height"]):
            raise CalibrationError(
                "ค่าสอบเทียบของ setup_id %s วัดที่ความละเอียด %dx%d "
                "แต่คลิปนี้เป็น %dx%d จึงใช้ร่วมกันไม่ได้"
                % (setup_id, cal["width"], cal["height"],
                   info["width"], info["height"]))
    return cal


def to_um(px, cal):
    return float(px) * float(cal["um_per_px"])


def detection_floor_um(cal, flow_resolution_px):
    return float(flow_resolution_px) * float(cal["um_per_px"])
