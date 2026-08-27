import io
import json
import os
import shutil
import sys
import time

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
REPO = os.path.dirname(APP)
for p in (REPO, APP):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("IRONPULSE_PROJECT_ROOT", APP)

from appcore import video, markers, calibrate
from web import baseline as bl
from web import db
from web import service as sv

cfg = sv.load_cfg()
sv.ensure_dirs(cfg)
FRAME_DIR = os.path.join(sv.abspath(cfg, "data_dir"), "frames")
if not os.path.isdir(FRAME_DIR):
    os.makedirs(FRAME_DIR)
CON = db.connect(sv.abspath(cfg, "db_file"))
CONFIG_HASH = bl.config_hash(sv.CONFIG_PATH)
CORE_VERSION = bl.core_version(REPO)

app = FastAPI(title="IronPulse Zero")
app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")
app.mount("/frames", StaticFiles(directory=FRAME_DIR), name="frames")
templates = Jinja2Templates(directory=os.path.join(HERE, "templates"))


def ctx(**kw):
    base = dict(cfg=cfg, config_hash=CONFIG_HASH,
                core_version=CORE_VERSION,
                min_clips=int(cfg["model"]["min_baseline_clips"]),
                detector=cfg["model"]["type"],
                default_fs=float(cfg["capture"]["default_fps"]),
                target_seconds=float(cfg["web"]["target_seconds"]),
                verdict_px=int(cfg["ui"]["verdict_font_px"]))
    base.update(kw)
    return base


def fail(msg, code=400):
    raise HTTPException(status_code=code, detail=msg)


def safe_clip_path(raw):
    if not raw:
        fail("ไม่ได้ระบุไฟล์คลิป")
    p = os.path.abspath(raw)
    allowed = [os.path.abspath(sv.abspath(cfg, k)) for k in ("demo_dir", "upload_dir")]
    if not any(p.startswith(a + os.sep) for a in allowed):
        fail("ไฟล์อยู่นอกโฟลเดอร์ที่อนุญาต ให้อัปโหลดไฟล์เข้ามาก่อน")
    if not os.path.isfile(p):
        fail("ไม่พบไฟล์ %s" % os.path.basename(p))
    if not sv.is_video(cfg, p):
        fail("ไฟล์นี้ไม่ใช่ไฟล์วิดีโอที่รองรับ นามสกุลที่รับคือ %s"
             % ", ".join(cfg["web"]["video_suffixes"]))
    return p


@app.exception_handler(Exception)
async def any_error(request, exc):
    if isinstance(exc, HTTPException):
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
    return JSONResponse({"detail": str(exc) or "เกิดข้อผิดพลาดที่ไม่คาดคิด"},
                        status_code=400)


@app.get("/", response_class=HTMLResponse)
def page_inspect(request: Request):
    return templates.TemplateResponse(request, "inspect.html", ctx(page="inspect"))


@app.get("/calibrate", response_class=HTMLResponse)
def page_calibrate(request: Request):
    return templates.TemplateResponse(request, "calibrate.html", ctx(page="calibrate"))


@app.get("/history", response_class=HTMLResponse)
def page_history(request: Request):
    return templates.TemplateResponse(request, "history.html", ctx(page="history"))


@app.get("/api/machines")
def api_machines():
    return {"machines": db.machines(CON)}


@app.get("/api/scales")
def api_scales():
    cals = calibrate.load(os.path.join(APP, cfg["scale"]["calibration_file"]))
    return {"scales": [{"id": k, "um_per_px": round(v["um_per_px"], 2),
                        "width": v["width"], "height": v["height"]}
                       for k, v in sorted(cals.items())]}


@app.get("/api/demo")
def api_demo():
    return {"files": [{"name": f["name"], "path": f["path"],
                       "mb": round(f["bytes"] / 1e6, 1)} for f in sv.demo_files(cfg)]}


@app.post("/api/upload")
async def api_upload(files: list[UploadFile] = File(...)):
    out = []
    limit = float(cfg["web"]["max_upload_mb"]) * 1e6
    for f in files:
        if not sv.is_video(cfg, f.filename or ""):
            fail("ไฟล์ %s ไม่ใช่ไฟล์วิดีโอที่รองรับ นามสกุลที่รับคือ %s"
                 % (f.filename, ", ".join(cfg["web"]["video_suffixes"])))
        dest = os.path.join(sv.abspath(cfg, "upload_dir"),
                            "%d_%s" % (int(time.time() * 1000), os.path.basename(f.filename)))
        n = 0
        with open(dest, "wb") as w:
            while True:
                chunk = await f.read(1 << 20)
                if not chunk:
                    break
                n += len(chunk)
                if n > limit:
                    w.close()
                    os.remove(dest)
                    fail("ไฟล์ %s ใหญ่เกิน %.0f MB" % (f.filename, limit / 1e6))
                w.write(chunk)
        out.append({"name": f.filename, "path": dest, "mb": round(n / 1e6, 1)})
    return {"files": out}


@app.post("/api/first-frame")
def api_first_frame(path: str = Form(...)):
    p = safe_clip_path(path)
    try:
        info = video.probe(p)
    except Exception as exc:
        fail(str(exc))
    name = "%s.png" % abs(hash(p + str(os.path.getmtime(p))))
    out = os.path.join(FRAME_DIR, name)
    if not os.path.isfile(out):
        sv.first_frame_png(p, out)
    return {"url": "/frames/" + name, "width": info["width"],
            "height": info["height"], "n_frames": info["n_frames"],
            "meta_fps": info["meta_fps"], "path": p}


@app.post("/api/machines")
def api_create_machine(name: str = Form(...), blade_count: str = Form(""),
                       fs: float = Form(...), scale_id: str = Form(""),
                       roi: str = Form(...)):
    pts = json.loads(roi)
    if len(pts) != 3:
        fail("ต้องคลิกมาร์กเกอร์ให้ครบ 3 จุด ตามลำดับ เครื่อง เสา อ้างอิง")
    bc = int(blade_count) if str(blade_count).strip() else None
    if db.one(CON, "SELECT id FROM machines WHERE name=?", (name,)):
        fail("มีเครื่องชื่อ %s อยู่แล้ว ให้ใช้ชื่ออื่นหรือเลือกเครื่องเดิม" % name)
    mid = db.add_machine(CON, name, bc, float(fs), pts, scale_id or None)
    return {"machine_id": mid}


@app.post("/api/machine/{machine_id}/roi")
def api_update_roi(machine_id: int, roi: str = Form(...), fs: float = Form(...),
                   blade_count: str = Form(""), scale_id: str = Form("")):
    m = db.machine(CON, machine_id)
    if not m:
        fail("ไม่พบเครื่องที่เลือก")
    pts = json.loads(roi)
    if len(pts) != 3:
        fail("ต้องคลิกมาร์กเกอร์ให้ครบ 3 จุด")
    bc = int(blade_count) if str(blade_count).strip() else m["blade_count"]
    db.update_machine_roi(CON, machine_id, pts, float(fs), bc, scale_id or m["scale_id"])
    return {"ok": True}


@app.post("/api/baseline")
def api_baseline(machine_id: int = Form(...), paths: str = Form(...)):
    m = db.machine(CON, machine_id)
    if not m:
        fail("ไม่พบเครื่องที่เลือก")
    files = [safe_clip_path(p) for p in json.loads(paths)]
    need = int(cfg["model"]["min_baseline_clips"])
    if len(files) < need:
        fail("มีคลิป %d คลิป ต้องการอย่างน้อย %d คลิป ขาดอีก %d คลิป"
             % (len(files), need, need - len(files)))
    jid = sv.new_job("baseline", total=len(files))

    def work():
        sess = sv.sessions_for(files, float(cfg["web"]["session_gap_sec"]))
        items = []
        for i, p in enumerate(files):
            sv.update(jid, step=i, message=os.path.basename(p))
            a = sv.analyse_one(cfg, p, m["roi"], float(m["fs"]), m["blade_count"],
                               progress=sv.progress_cb(jid))
            items.append(dict(vector=a["vector"], names=a["names"],
                              primary_axis=a["primary_axis"], f0_hz=a["f0_hz"],
                              fs=a["fs"], spectrum=a["spectrum"],
                              blade_count=m["blade_count"]))
        sv.update(jid, step=len(files), message="สร้างแบบจำลอง")
        payload = bl.build(items, cfg, m["scale_id"], CONFIG_HASH, CORE_VERSION, sess)
        bid = db.add_baseline(CON, machine_id, payload)
        return {"baseline_id": bid, "n_clips": payload["n_clips"],
                "f0_hz": payload["f0_hz"], "threshold": payload["threshold"],
                "primary_axis": payload["primary_axis"],
                "detector": payload["detector_type"],
                "feature_set": payload["feature_set"],
                "n_features": len(payload["feature_names"]),
                "sessions": sorted(set(sess)), "warnings": payload["warnings"]}

    sv.run_bg(jid, work)
    return {"job": jid}


@app.post("/api/inspect")
def api_inspect(machine_id: int = Form(...), path: str = Form(...),
                fs: str = Form("")):
    m = db.machine(CON, machine_id)
    if not m:
        fail("ไม่พบเครื่องที่เลือก")
    p = safe_clip_path(path)
    override = float(fs) if str(fs).strip() else None
    jid = sv.new_job("inspect")

    def work():
        return sv.inspect(cfg, CON, m, p, os.path.basename(p), fs=override,
                          progress=sv.progress_cb(jid))

    sv.run_bg(jid, work)
    return {"job": jid}


@app.get("/api/job/{jid}")
def api_job(jid: str):
    j = sv.job(jid)
    if not j:
        fail("ไม่พบงานที่ร้องขอ อาจหมดอายุแล้ว ให้กดวิเคราะห์ใหม่", 404)
    return j


@app.get("/api/history/{machine_id}")
def api_history(machine_id: int):
    return {"rows": db.inspections(CON, machine_id)}


@app.get("/api/inspection/{inspection_id}")
def api_inspection(inspection_id: int):
    r = db.inspection(CON, inspection_id)
    if not r:
        fail("ไม่พบผลการตรวจนี้", 404)
    return r


@app.get("/api/history/{machine_id}/csv", response_class=PlainTextResponse)
def api_history_csv(machine_id: int):
    rows = db.inspections(CON, machine_id, limit=100000)
    if not rows:
        return "no rows\n"
    keys = list(rows[0].keys())
    out = io.StringIO()
    out.write(",".join(keys) + "\n")
    for r in rows:
        out.write(",".join("" if r[k] is None else str(r[k]) for k in keys) + "\n")
    return out.getvalue()


@app.get("/api/health")
def api_health():
    return {"config_hash": CONFIG_HASH, "core_version": CORE_VERSION,
            "detector": cfg["model"]["type"],
            "feature_set": cfg["model"]["feature_set"],
            "fs_default": float(cfg["capture"]["default_fps"]),
            "min_baseline_clips": int(cfg["model"]["min_baseline_clips"])}
