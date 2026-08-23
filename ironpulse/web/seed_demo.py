import csv
import io
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
REPO = os.path.dirname(APP)
for p in (REPO, APP):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("IRONPULSE_PROJECT_ROOT", APP)

from web import baseline as bl
from web import db
from web import service as sv

INV = os.path.join(APP, "h4", "inventory.csv")
META = os.path.join(APP, "h4", "tracks_meta.csv")
MACHINE_NAME = "พัดลมตั้งพื้น H4"
SCALE_ID = "h4_wall_2026_08_22"
HOLDOUT_NORMAL = 2
DEMO_FAULT = {"3g": 1, "under3g": 1}
LINE = "=" * 90


def main():
    cfg = sv.load_cfg()
    sv.ensure_dirs(cfg)
    inv = list(csv.DictReader(io.open(INV, encoding="utf-8")))
    meta = {r["clip"]: r for r in csv.DictReader(io.open(META, encoding="utf-8"))}
    by_group = {}
    for r in inv:
        by_group.setdefault(r["group"], []).append(r)
    for g in by_group:
        by_group[g].sort(key=lambda r: r["clip"])

    normals = by_group["normal"]
    holdout = normals[-HOLDOUT_NORMAL:]
    train = normals[:-HOLDOUT_NORMAL]
    demo_rows = list(holdout)
    for g, n in DEMO_FAULT.items():
        demo_rows += by_group[g][:n]

    demo_dir = sv.abspath(cfg, "demo_dir")
    print(LINE)
    print("คัดลอกคลิปสาธิตไปที่ %s" % demo_dir)
    print(LINE)
    for r in demo_rows:
        label = {"normal": "normal", "3g": "unbal3g", "under3g": "unbal_low"}[r["group"]]
        dest = os.path.join(demo_dir, "%s_%s.MOV" % (label, r["clip"]))
        if not os.path.isfile(dest):
            shutil.copy2(r["path"], dest)
        print("  %-28s <- %s (%s)" % (os.path.basename(dest), r["clip"], r["group"]))
    print("  คลิปปกติสองไฟล์นี้ถูกกันออกจาก baseline จึงเป็นคลิปที่โมเดลไม่เคยเห็น")

    con = db.connect(sv.abspath(cfg, "db_file"))
    m = db.one(con, "SELECT * FROM machines WHERE name=?", (MACHINE_NAME,))
    first = meta[train[0]["clip"]]
    roi = [[int(first["seed_machine_x"]), int(first["seed_machine_y"])],
           [int(first["seed_base_x"]), int(first["seed_base_y"])],
           [int(first["seed_ref_x"]), int(first["seed_ref_y"])]]
    fs = float(cfg["capture"]["default_fps"])
    blades = cfg["machine"]["blade_count"]
    if m is None:
        mid = db.add_machine(con, MACHINE_NAME, blades, fs, roi, SCALE_ID,
                             "ตั้งค่าจากชุด Arise H4")
        print("\nสร้างเครื่อง %s (id %d)" % (MACHINE_NAME, mid))
    else:
        mid = m["id"]
        db.update_machine_roi(con, mid, roi, fs, blades, SCALE_ID)
        print("\nใช้เครื่องเดิม %s (id %d)" % (MACHINE_NAME, mid))
    machine = db.machine(con, mid)

    if db.active_baseline(con, mid):
        print("มี baseline อยู่แล้ว ไม่สร้างซ้ำ ถ้าต้องการสร้างใหม่ให้ลบไฟล์ฐานข้อมูลก่อน")
        return 0

    print()
    print(LINE)
    print("สร้าง baseline จากคลิปปกติ %d คลิป ผ่านเส้นทางเดียวกับเว็บแอป" % len(train))
    print(LINE)
    paths = [r["path"] for r in train]
    sess = sv.sessions_for(paths, float(cfg["web"]["session_gap_sec"]))
    items = []
    for i, r in enumerate(train):
        mrow = meta[r["clip"]]
        roi_i = [(int(mrow["seed_machine_x"]), int(mrow["seed_machine_y"])),
                 (int(mrow["seed_base_x"]), int(mrow["seed_base_y"])),
                 (int(mrow["seed_ref_x"]), int(mrow["seed_ref_y"]))]
        a = sv.analyse_one(cfg, r["path"], roi_i, fs, blades)
        items.append(dict(vector=a["vector"], names=a["names"],
                          primary_axis=a["primary_axis"], f0_hz=a["f0_hz"],
                          fs=a["fs"], spectrum=a["spectrum"], blade_count=blades))
        print("  %2d/%d %-11s f0 %.3f Hz  %.1fs" %
              (i + 1, len(train), r["clip"], a["f0_hz"], a["seconds"]), flush=True)
    payload = bl.build(items, cfg, SCALE_ID, bl.config_hash(sv.CONFIG_PATH),
                       bl.core_version(REPO), sess)
    bid = db.add_baseline(con, mid, payload)
    print()
    print("baseline id %d · %d คลิป · %s · ชุด feature %s (%d ตัว) · แกน %s"
          % (bid, payload["n_clips"], payload["detector_type"],
             payload["feature_set"], len(payload["feature_names"]),
             payload["primary_axis"]))
    print("f0 %.4f Hz · เกณฑ์ %.6f · รอบการถ่าย %s"
          % (payload["f0_hz"], payload["threshold"], sorted(set(sess))))
    for w in payload["warnings"]:
        print("  ! " + w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
