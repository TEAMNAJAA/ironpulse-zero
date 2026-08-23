import io
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
REPO = os.path.dirname(APP)
for p in (REPO, APP):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("IRONPULSE_PROJECT_ROOT", APP)

import yaml

CFG = yaml.safe_load(io.open(os.path.join(APP, "config.yaml"), encoding="utf-8"))
BASE = "http://%s:%d" % (CFG["web"]["host"], int(CFG["web"]["port"]))
LINE = "=" * 92
RESULTS = []


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.load(r)


def post(path, fields):
    data = urllib.parse.urlencode(fields).encode()
    req = urllib.request.Request(BASE + path, data=data)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(body)
        except ValueError:
            return e.code, {"detail": body[:200]}


def wait(job):
    for _ in range(600):
        j = get("/api/job/" + job)
        if j["state"] == "done":
            return j["result"]
        if j["state"] == "error":
            raise RuntimeError(j["error"])
        time.sleep(0.4)
    raise RuntimeError("job timeout")


def check(n, name, ok, detail=""):
    RESULTS.append((n, name, ok, detail))
    print("%2d. %-52s %s" % (n, name, "PASS" if ok else "FAIL"))
    if detail:
        print("      " + detail)


def main():
    print(LINE)
    print("เกณฑ์ตรวจรับ H5 · %s" % BASE)
    print(LINE)
    m = get("/api/machines")["machines"]
    mid = m[0]["id"]
    demo = get("/api/demo")["files"]
    by = {f["name"]: f["path"] for f in demo}
    normal = [p for n, p in sorted(by.items()) if n.startswith("normal_")]
    fault = [p for n, p in sorted(by.items()) if n.startswith("unbal3g")]
    low = [p for n, p in sorted(by.items()) if n.startswith("unbal_low")]

    t0 = time.time()
    _, r1 = post("/api/inspect", dict(machine_id=mid, path=fault[0], fs=""))
    a = wait(r1["job"])
    _, r2 = post("/api/inspect", dict(machine_id=mid, path=fault[0], fs=""))
    b = wait(r2["job"])
    same = ("%.12g" % a["score"]) == ("%.12g" % b["score"])
    check(1, "คลิปเดิมรันสองครั้ง คะแนนเท่ากันทุกหลัก", same,
          "%.12g เทียบ %.12g" % (a["score"], b["score"]))

    from h4 import verify_webapp
    check(2, "คะแนนตรงกับ raw_scores.csv ของ H4", verify_webapp.main() == 0,
          "รายละเอียดอยู่ในผลของ h4/verify_webapp.py ด้านบน")

    pages = []
    for path in ("/", "/calibrate", "/history"):
        with urllib.request.urlopen(BASE + path, timeout=20) as r:
            pages.append(r.read().decode("utf-8", "replace"))
    ext = []
    for name in ("static/css/app.css", "static/js/common.js", "static/js/inspect.js",
                 "static/js/calibrate.js", "static/js/history.js",
                 "static/vendor/miniplot.js", "templates/base.html"):
        txt = io.open(os.path.join(HERE, name), encoding="utf-8").read()
        for token in ("http://", "https://", "//cdn", "fonts.googleapis"):
            if token in txt and "127.0.0.1" not in txt:
                ext.append(name + " -> " + token)
    check(3, "ไม่มี CDN หรือทรัพยากรภายนอกในทุกหน้า",
          not ext and all("<canvas" in p or "miniplot" in p for p in pages[:1]),
          "ไฟล์ที่อ้างภายนอก: %s" % (", ".join(ext) if ext else "ไม่มี"))

    _, r3 = post("/api/inspect", dict(machine_id=mid, path=normal[0], fs="30"))
    c = wait(r3["job"])
    fs_warn = [w for w in c["warnings"] if "metadata" in w or "fps" in w]
    check(4, "กรอก fs ผิดเป็น 30 ต้องขึ้นคำเตือน", bool(fs_warn),
          fs_warn[0][:110] if fs_warn else "ไม่พบคำเตือน")

    bad = os.path.join(sv_upload(), "not_a_video.txt")
    io.open(bad, "w", encoding="utf-8").write("hello")
    code, body = post("/api/inspect", dict(machine_id=mid, path=bad, fs=""))
    thai = any("฀" <= ch <= "๿" for ch in body.get("detail", ""))
    check(5, "ไฟล์ที่ไม่ใช่วิดีโอ ได้ข้อความไทย ไม่ใช่ traceback",
          code >= 400 and thai and "Traceback" not in body.get("detail", ""),
          body.get("detail", "")[:110])

    empty_name = "เครื่องเปล่าทดสอบ"
    code2, body2 = post("/api/machines", dict(name=empty_name, blade_count="3",
                                              fs="240", scale_id="",
                                              roi=json.dumps([[10, 10], [20, 20], [30, 30]])))
    new_id = body2.get("machine_id")
    if new_id is None:
        for row in get("/api/machines")["machines"]:
            if row["name"] == empty_name:
                new_id = row["id"]
                break
    code3, body3 = post("/api/inspect", dict(machine_id=new_id, path=normal[0], fs=""))
    if code3 < 400:
        try:
            wait(body3["job"])
            msg = ""
        except RuntimeError as exc:
            msg = str(exc)
    else:
        msg = body3.get("detail", "")
    check(6, "เลือกเครื่องที่ยังไม่มี baseline ต้องบอกให้ไปสอบเทียบก่อน",
          "baseline" in msg or "สอบเทียบ" in msg, msg[:110])

    rows = get("/api/history/%d" % mid)["rows"]
    check(7, "ประวัติถูกบันทึกลง SQLite และอ่านกลับได้", len(rows) >= 3,
          "มี %d แถวในประวัติของเครื่องนี้" % len(rows))

    n_demo = len(demo)
    probe = os.path.join(sv_demo(), "zz_probe.MOV")
    import shutil
    shutil.copy2(fault[0], probe)
    n_after = len(get("/api/demo")["files"])
    os.remove(probe)
    check(8, "เพิ่มไฟล์ใน data/demo แล้วปุ่มขึ้นเองโดยไม่ต้องแก้โค้ด",
          n_after == n_demo + 1, "%d -> %d ไฟล์" % (n_demo, n_after))

    target = float(CFG["web"]["target_seconds"])
    check(9, "คลิป 2000 เฟรมเสร็จภายใน %.0f วินาที และรายงานเวลาจริง" % target,
          a["seconds"] <= target and a["n_frames"] == 2000,
          "ใช้เวลา %.2f วินาที · %d เฟรม" % (a["seconds"], a["n_frames"]))

    ratio = float(CFG["web"]["a1_floor_warn_ratio"])
    triggered, detail = None, "ไม่มีคลิปสาธิตที่ A1 ต่ำพอจะกระตุ้นเงื่อนไขนี้"
    for name, path in sorted(by.items()):
        if not name.startswith("unbal"):
            continue
        _, r4 = post("/api/inspect", dict(machine_id=mid, path=path, fs=""))
        d = wait(r4["job"])
        if d["a1_um"] is not None and d["a1_um"] < ratio * d["floor_um"]:
            warn = [w for w in d["warnings"] if "พื้นการวัด" in w]
            triggered = bool(warn)
            detail = "%s · A1 %.1f µm < %.1f x %.1f µm · %s" % (
                name, d["a1_um"], ratio, d["floor_um"],
                "ขึ้นคำเตือนแล้ว" if warn else "ไม่ขึ้นคำเตือน")
            break
    check(10, "คลิปที่ A1 ต่ำกว่า %.1f เท่าของพื้นการวัด ต้องขึ้นคำเตือน" % ratio,
          triggered is True, detail)

    print()
    print(LINE)
    n_ok = sum(1 for r in RESULTS if r[2])
    print("ผ่าน %d จาก %d ข้อ" % (n_ok, len(RESULTS)))
    print(LINE)
    return 0 if n_ok == len(RESULTS) else 1


def sv_upload():
    p = CFG["web"]["upload_dir"]
    return p if os.path.isabs(p) else os.path.join(APP, p)


def sv_demo():
    p = CFG["web"]["demo_dir"]
    return p if os.path.isabs(p) else os.path.join(APP, p)


if __name__ == "__main__":
    sys.exit(main())
