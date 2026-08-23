import os
import sys
import io
import csv
import yaml
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
REPO = os.path.dirname(APP)
sys.path.insert(0, REPO)
sys.path.insert(0, APP)
os.environ.setdefault("IRONPULSE_PROJECT_ROOT", APP)

from core import features as F
from appcore import markers, cfgmap

NPZ_IN = os.path.join(HERE, "tracks.npz")
META = os.path.join(HERE, "tracks_meta.csv")
NPZ_OUT = os.path.join(HERE, "features.npz")
CSV_OUT = os.path.join(HERE, "features_full.csv")
WIN = 1024
N_WIN = 3


def window_starts(n, win, k):
    if n <= win:
        return [0]
    hop = (n - win) // (k - 1)
    return [i * hop for i in range(k)]


def vector_for(sigs, fs, f0, ccfg, app_cfg):
    names, vals, avail = [], [], {}
    for sname in app_cfg["features"]["model_signals"]:
        if sname not in sigs:
            continue
        feat, raw, av = F.extract(sigs[sname], fs, f0, ccfg)
        for k in app_cfg["features"]["keep"]:
            names.append("%s.%s" % (sname, k))
            vals.append(float(feat[k]))
            avail["%s.%s" % (sname, k)] = bool(av[k])
    return np.array(vals, dtype=np.float64), names, avail


def main():
    cfg = yaml.safe_load(io.open(os.path.join(APP, "config.yaml"), encoding="utf-8"))
    fs = float(cfg["capture"]["default_fps"])
    blades = cfgmap.rbp_blade_count(cfg, cfg["machine"]["blade_count"])
    meta = list(csv.DictReader(io.open(META, encoding="utf-8")))
    d = np.load(NPZ_IN)
    clips = [r["clip"] for r in meta]
    by = {r["clip"]: r for r in meta}

    full_rows, full_vecs = [], []
    win_vecs, win_clip, win_index = [], [], []
    names_ref, avail_ref = None, None
    for c in clips:
        t = d[c].astype(np.float64)
        f0 = float(by[c]["f0_hz"])
        ccfg = cfgmap.core_cfg(cfg, fs, f0, blades)
        sigs = markers.differential(t)
        v, names, avail = vector_for(sigs, fs, f0, ccfg, cfg)
        names_ref, avail_ref = names, avail
        full_vecs.append(v)
        row = dict(clip=c, group=by[c]["group"], severity=by[c]["severity"],
                   session=by[c]["session"], f0_hz=f0,
                   primary_axis=by[c]["primary_axis"])
        for nm, val in zip(names, v):
            row[nm] = round(float(val), 8)
        full_rows.append(row)
        n = t.shape[0]
        for wi, s0 in enumerate(window_starts(n, WIN, N_WIN)):
            tw = t[s0:s0 + WIN]
            sw = markers.differential(tw)
            vw, _, _ = vector_for(sw, fs, f0, ccfg, cfg)
            win_vecs.append(vw)
            win_clip.append(c)
            win_index.append(wi)
        print("%-10s %-8s f0=%7.3f  full+%d windows" % (c, by[c]["group"], f0, N_WIN),
              flush=True)

    V = np.vstack(full_vecs)
    W = np.vstack(win_vecs)
    zero = [names_ref[j] for j in range(V.shape[1]) if V[:, j].std() == 0.0]
    unavail = sorted(set(k for k, ok in avail_ref.items() if not ok))
    np.savez_compressed(
        NPZ_OUT,
        vectors=V, names=np.array(names_ref, dtype=object),
        clips=np.array(clips, dtype=object),
        groups=np.array([by[c]["group"] for c in clips], dtype=object),
        severity=np.array([int(by[c]["severity"]) for c in clips]),
        sessions=np.array([by[c]["session"] for c in clips], dtype=object),
        f0=np.array([float(by[c]["f0_hz"]) for c in clips]),
        primary=np.array([by[c]["primary_axis"] for c in clips], dtype=object),
        win_vectors=W, win_clip=np.array(win_clip, dtype=object),
        win_index=np.array(win_index))
    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(full_rows[0].keys()))
        w.writeheader()
        w.writerows(full_rows)
    print()
    print("clip level    :", V.shape)
    print("window level  :", W.shape, "(%d windows of %d frames per clip)" % (N_WIN, WIN))
    print("features not measured (avail False):", ", ".join(unavail) or "none")
    print("features with zero variance across all clips:", ", ".join(zero) or "none")
    print("wrote", NPZ_OUT)
    print("wrote", CSV_OUT)


if __name__ == "__main__":
    main()
