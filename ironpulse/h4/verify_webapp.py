import io
import os
import sys
import csv

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
REPO = os.path.dirname(APP)
for p in (REPO, APP):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("IRONPULSE_PROJECT_ROOT", APP)

from web import baseline as bl
from web import service as sv

NPZ = os.path.join(HERE, "features.npz")
META = os.path.join(HERE, "tracks_meta.csv")
RAW = os.path.join(HERE, "results", "raw_scores.csv")
INV = os.path.join(HERE, "inventory.csv")
FOLD = "S2"
SEED = 42
LEVEL = "clip"
N_PIPELINE_CLIPS = 3
LINE = "=" * 96


def main():
    cfg = sv.load_cfg()
    meta = {r["clip"]: r for r in csv.DictReader(io.open(META, encoding="utf-8"))}
    inv = {r["clip"]: r for r in csv.DictReader(io.open(INV, encoding="utf-8"))}
    d = np.load(NPZ, allow_pickle=True)
    names = [str(x) for x in d["names"]]
    clips = [str(x) for x in d["clips"]]
    groups = [str(x) for x in d["groups"]]
    sessions = [str(x) for x in d["sessions"]]
    V = d["vectors"]
    idx = {c: i for i, c in enumerate(clips)}

    print(LINE)
    print("A. pipeline equivalence: web analyse_one vs h4 features.npz")
    print(LINE)
    picked = []
    for g in ("normal", "3g", "under3g"):
        for c in clips:
            if groups[idx[c]] == g and c not in picked:
                picked.append(c)
                break
    picked = picked[:N_PIPELINE_CLIPS]
    worst = 0.0
    for c in picked:
        m = meta[c]
        roi = [(int(m["seed_machine_x"]), int(m["seed_machine_y"])),
               (int(m["seed_base_x"]), int(m["seed_base_y"])),
               (int(m["seed_ref_x"]), int(m["seed_ref_y"]))]
        a = sv.analyse_one(cfg, inv[c]["path"], roi, float(cfg["capture"]["default_fps"]),
                           cfg["machine"]["blade_count"], baseline_f0=None)
        ref = V[idx[c]]
        if list(a["names"]) != names:
            print("  %-11s FEATURE NAMES DIFFER" % c)
            continue
        diff = float(np.max(np.abs(a["vector"] - ref)))
        worst = max(worst, diff)
        print("  %-11s %-8s f0 %.4f vs %.4f   max |diff| over 39 features = %.3e   %s"
              % (c, groups[idx[c]], a["f0_hz"], float(m["f0_hz"]), diff,
                 "MATCH" if diff == 0.0 else "DIFFER"))
    print()
    print("  worst absolute difference across %d clips: %.3e" % (len(picked), worst))

    print()
    print(LINE)
    print("B. detector equivalence: web baseline.build vs h4 fold %s seed %d" % (FOLD, SEED))
    print(LINE)
    tr = [c for c in clips if groups[idx[c]] == "normal" and sessions[idx[c]] != FOLD]
    te = [c for c in clips if groups[idx[c]] == "normal" and sessions[idx[c]] == FOLD]
    fault = [c for c in clips if groups[idx[c]] != "normal"]
    items = []
    for c in tr:
        items.append(dict(vector=V[idx[c]], names=names,
                          primary_axis=meta[c]["primary_axis"],
                          f0_hz=float(meta[c]["f0_hz"]),
                          fs=float(cfg["capture"]["default_fps"]),
                          spectrum=[0.0], blade_count=cfg["machine"]["blade_count"]))
    payload = bl.build(items, cfg, None, "verify", "verify",
                       [sessions[idx[c]] for c in tr])
    base = dict(payload)
    det = bl.refit(cfg, base, cfg["machine"]["blade_count"])
    thr = float(base["threshold"])

    raw = [r for r in csv.DictReader(io.open(RAW, encoding="utf-8"))
           if r["level"] == LEVEL and r["fold"] == FOLD and int(r["seed"]) == SEED]
    ref_score = {r["clip"]: float(r["score"]) for r in raw}
    ref_thr = float(raw[0]["threshold"])
    print("  train %d clips, feature set %s, %d features, axis %s"
          % (len(tr), base["feature_set"], len(base["feature_names"]),
             payload["primary_axis"]))
    print("  threshold web %.10f   h4 %.10f   diff %.3e"
          % (thr, ref_thr, abs(thr - ref_thr)))
    print()
    print("  %-11s %-8s %14s %14s %12s" % ("clip", "group", "web score", "h4 score", "diff"))
    worst2 = 0.0
    n = 0
    for c in te + fault:
        if c not in ref_score:
            continue
        s = bl.score_vector(cfg, base, det, V[idx[c]], names)
        diff = abs(s - ref_score[c])
        worst2 = max(worst2, diff)
        n += 1
        if n <= 12:
            print("  %-11s %-8s %14.8f %14.8f %12.2e" %
                  (c, groups[idx[c]], s, ref_score[c], diff))
    print("  ... %d clips compared in total" % n)
    print()
    print("  worst absolute score difference: %.3e" % worst2)
    print()
    print(LINE)
    tol = 5e-9
    print("  raw_scores.csv keeps 8 decimals, so the comparison tolerance is %.0e" % tol)
    ok = worst == 0.0 and worst2 <= tol and abs(thr - ref_thr) <= tol
    print("M1 acceptance: %s" % ("PASS" if ok else "FAIL"))
    print(LINE)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
