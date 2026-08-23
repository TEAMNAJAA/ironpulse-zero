import io
import json
import os
import sqlite3
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS machines(
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  blade_count INTEGER,
  fs REAL NOT NULL,
  roi_json TEXT NOT NULL,
  scale_id TEXT,
  notes TEXT,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS baselines(
  id INTEGER PRIMARY KEY,
  machine_id INTEGER NOT NULL,
  fs REAL NOT NULL,
  f0_hz REAL NOT NULL,
  n_clips INTEGER NOT NULL,
  detector_type TEXT NOT NULL,
  feature_set TEXT NOT NULL,
  feature_names_json TEXT NOT NULL,
  train_vectors_json TEXT NOT NULL,
  feature_mean_json TEXT NOT NULL,
  feature_std_json TEXT NOT NULL,
  threshold REAL NOT NULL,
  seed INTEGER NOT NULL,
  spectrum_json TEXT NOT NULL,
  sessions_json TEXT NOT NULL,
  config_hash TEXT NOT NULL,
  core_version TEXT NOT NULL,
  scale_id TEXT,
  warnings_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  is_active INTEGER DEFAULT 1
);
CREATE TABLE IF NOT EXISTS inspections(
  id INTEGER PRIMARY KEY,
  machine_id INTEGER NOT NULL,
  baseline_id INTEGER,
  filename TEXT NOT NULL,
  fs REAL NOT NULL,
  f0_hz REAL,
  n_frames INTEGER,
  score REAL,
  threshold REAL,
  ratio REAL,
  verdict TEXT,
  a1_px REAL,
  a1_um REAL,
  floor_um REAL,
  primary_axis TEXT,
  seconds REAL,
  detector_type TEXT NOT NULL,
  config_hash TEXT NOT NULL,
  core_version TEXT NOT NULL,
  scale_id TEXT,
  warnings_json TEXT NOT NULL,
  result_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_insp_machine ON inspections(machine_id, created_at);
"""


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def connect(path):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.isdir(d):
        os.makedirs(d)
    con = sqlite3.connect(path, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    return con


def rows(con, sql, args=()):
    return [dict(r) for r in con.execute(sql, args).fetchall()]


def one(con, sql, args=()):
    r = con.execute(sql, args).fetchone()
    return dict(r) if r else None


def add_machine(con, name, blade_count, fs, roi, scale_id, notes=""):
    cur = con.execute(
        "INSERT INTO machines(name, blade_count, fs, roi_json, scale_id, notes, created_at)"
        " VALUES(?,?,?,?,?,?,?)",
        (name, blade_count, fs, json.dumps(roi), scale_id, notes, now()))
    con.commit()
    return cur.lastrowid


def update_machine_roi(con, machine_id, roi, fs, blade_count, scale_id):
    con.execute("UPDATE machines SET roi_json=?, fs=?, blade_count=?, scale_id=?"
                " WHERE id=?",
                (json.dumps(roi), fs, blade_count, scale_id, machine_id))
    con.commit()


def machines(con):
    out = rows(con, "SELECT * FROM machines ORDER BY name")
    for m in out:
        m["roi"] = json.loads(m["roi_json"])
        b = active_baseline(con, m["id"])
        m["has_baseline"] = b is not None
        m["baseline_clips"] = b["n_clips"] if b else 0
    return out


def machine(con, machine_id):
    m = one(con, "SELECT * FROM machines WHERE id=?", (machine_id,))
    if m:
        m["roi"] = json.loads(m["roi_json"])
    return m


def add_baseline(con, machine_id, payload):
    con.execute("UPDATE baselines SET is_active=0 WHERE machine_id=?", (machine_id,))
    cur = con.execute(
        "INSERT INTO baselines(machine_id, fs, f0_hz, n_clips, detector_type,"
        " feature_set, feature_names_json, train_vectors_json, feature_mean_json,"
        " feature_std_json, threshold, seed, spectrum_json, sessions_json,"
        " config_hash, core_version, scale_id, warnings_json, created_at, is_active)"
        " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)",
        (machine_id, payload["fs"], payload["f0_hz"], payload["n_clips"],
         payload["detector_type"], payload["feature_set"],
         json.dumps(payload["feature_names"]), json.dumps(payload["train_vectors"]),
         json.dumps(payload["feature_mean"]), json.dumps(payload["feature_std"]),
         payload["threshold"], payload["seed"], json.dumps(payload["spectrum"]),
         json.dumps(payload["sessions"]), payload["config_hash"],
         payload["core_version"], payload["scale_id"],
         json.dumps(payload["warnings"]), now()))
    con.commit()
    return cur.lastrowid


def active_baseline(con, machine_id):
    b = one(con, "SELECT * FROM baselines WHERE machine_id=? AND is_active=1"
                 " ORDER BY id DESC LIMIT 1", (machine_id,))
    if not b:
        return None
    for k in ("feature_names", "train_vectors", "feature_mean", "feature_std",
              "spectrum", "sessions", "warnings"):
        b[k] = json.loads(b[k + "_json"])
    return b


def add_inspection(con, payload):
    cur = con.execute(
        "INSERT INTO inspections(machine_id, baseline_id, filename, fs, f0_hz,"
        " n_frames, score, threshold, ratio, verdict, a1_px, a1_um, floor_um,"
        " primary_axis, seconds, detector_type, config_hash, core_version,"
        " scale_id, warnings_json, result_json, created_at)"
        " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (payload["machine_id"], payload["baseline_id"], payload["filename"],
         payload["fs"], payload["f0_hz"], payload["n_frames"], payload["score"],
         payload["threshold"], payload["ratio"], payload["verdict"],
         payload["a1_px"], payload["a1_um"], payload["floor_um"],
         payload["primary_axis"], payload["seconds"], payload["detector_type"],
         payload["config_hash"], payload["core_version"], payload["scale_id"],
         json.dumps(payload["warnings"], ensure_ascii=False),
         json.dumps(payload["result"], ensure_ascii=False), now()))
    con.commit()
    return cur.lastrowid


def inspections(con, machine_id, limit=500):
    out = rows(con, "SELECT id, machine_id, filename, fs, f0_hz, n_frames, score,"
                    " threshold, ratio, verdict, a1_px, a1_um, floor_um,"
                    " primary_axis, seconds, detector_type, scale_id, created_at,"
                    " warnings_json FROM inspections WHERE machine_id=?"
                    " ORDER BY id DESC LIMIT ?", (machine_id, limit))
    for r in out:
        r["n_warnings"] = len(json.loads(r.pop("warnings_json")))
    return out


def inspection(con, inspection_id):
    r = one(con, "SELECT * FROM inspections WHERE id=?", (inspection_id,))
    if r:
        r["warnings"] = json.loads(r["warnings_json"])
        r["result"] = json.loads(r["result_json"])
    return r
