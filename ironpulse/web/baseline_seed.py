import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
REPO = os.path.dirname(APP)
for p in (REPO, APP):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("IRONPULSE_PROJECT_ROOT", APP)

from web import db
from web import service as sv

SEED = os.path.join(HERE, "seed", "baseline_seed.json")
KEEP = ("fs", "f0_hz", "n_clips", "detector_type", "feature_set", "feature_names",
        "train_vectors", "feature_mean", "feature_std", "threshold", "seed",
        "spectrum", "sessions", "config_hash", "core_version", "scale_id",
        "warnings")


def export(machine_name=None):
    cfg = sv.load_cfg()
    con = db.connect(sv.abspath(cfg, "db_file"))
    machines = db.machines(con)
    if not machines:
        raise SystemExit("ไม่มีเครื่องในฐานข้อมูล ให้รัน web/seed_demo.py ก่อน")
    picked = None
    for m in machines:
        if machine_name and m["name"] != machine_name:
            continue
        if m["has_baseline"]:
            picked = m
            break
    if picked is None:
        raise SystemExit("ไม่มีเครื่องที่มี baseline ให้ส่งออก")
    base = db.active_baseline(con, picked["id"])
    payload = dict(
        machine=dict(name=picked["name"], blade_count=picked["blade_count"],
                     fs=picked["fs"], roi=picked["roi"], scale_id=picked["scale_id"],
                     notes=picked["notes"] or ""),
        baseline={k: base[k] for k in KEEP})
    os.makedirs(os.path.dirname(SEED), exist_ok=True)
    io.open(SEED, "w", encoding="utf-8", newline="\n").write(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    print("wrote %s  %.0f KB" % (SEED, os.path.getsize(SEED) / 1e3))
    print("machine  : %s" % picked["name"])
    print("baseline : %d คลิป · %s · %s · เกณฑ์ %.6f"
          % (base["n_clips"], base["detector_type"], base["feature_set"],
             base["threshold"]))
    return SEED


def load_into(con, cfg):
    if not os.path.isfile(SEED):
        return None
    if db.machines(con):
        return None
    payload = json.loads(io.open(SEED, encoding="utf-8").read())
    m = payload["machine"]
    mid = db.add_machine(con, m["name"], m["blade_count"], m["fs"], m["roi"],
                         m["scale_id"], m["notes"])
    db.add_baseline(con, mid, payload["baseline"])
    return dict(machine_id=mid, name=m["name"],
                n_clips=payload["baseline"]["n_clips"])


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "import":
        cfg = sv.load_cfg()
        sv.ensure_dirs(cfg)
        con = db.connect(sv.abspath(cfg, "db_file"))
        out = load_into(con, cfg)
        if out is None:
            existing = db.machines(con)
            if existing:
                print("ฐานข้อมูลมีเครื่องอยู่แล้ว %d เครื่อง ไม่นำเข้าซ้ำ" % len(existing))
            else:
                print("ไม่พบไฟล์ %s จึงไม่ได้นำเข้าอะไร" % SEED)
        else:
            print("นำเข้าเครื่อง %s พร้อม baseline %d คลิป"
                  % (out["name"], out["n_clips"]))
        return 0
    export(sys.argv[1] if len(sys.argv) > 1 else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
