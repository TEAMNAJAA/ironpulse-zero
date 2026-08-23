import os, sys, copy
import numpy as np
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)
sys.path.insert(0, os.path.dirname(_PROJ))
from experiments.common import CFG, maf_index_all, write_csv, table
from core import pipeline, camera_sim as cs, features as F, detectors, evaluate as EV
from core import signals as S

FS = float(CFG["camera"]["primary_fs_hz"])
EXP = CFG["camera"]["default_exposure_sec"]
GROUPS = ["orders", "bands", "shape", "time"]
MAX_F0_FOR_3X = (FS / 2.0) / 3.0


def build(index):
    out = []
    for rec in index:
        sig, fsb, m = pipeline.signal_for(rec, CFG, domain="displacement")
        if sig is None:
            continue
        y = cs.decimate(sig, fsb, FS, mode="boxcar", exposure_sec=EXP)
        y = cs.to_pixels(y, CFG["optics"]["pixels_per_meter"])
        out.append((y, m["f0_hz"], m["label"], m["file_id"]))
    return out


def featurise(sigs):
    X, meta = [], []
    for y, f0, lab, fid in sigs:
        for s0, w in S.windows(y, FS, CFG["signal"]["window_sec"], CFG["signal"]["hop_sec"]):
            f, raw, av = F.extract(w, FS, f0, CFG)
            X.append(F.vector(f))
            meta.append(dict(file_id=fid, label=lab, f0_hz=f0))
    return np.vstack(X), meta


def subset(X, names):
    idx = [F.DIMENSIONLESS.index(n) for n in names]
    return X[:, idx]


def evaluate(X, meta, tag, variant, rows):
    r, _ = EV.run_cv(X, meta, detectors.MahalanobisDetector, CFG, per_label=False)
    if not r:
        return None
    a = EV.agg(r)
    a.update(population=tag, variant=variant, n_windows=len(meta),
             n_features=X.shape[1])
    rows.append(a)
    return a["roc_auc_mean"]


if __name__ == "__main__":
    sigs = build(maf_index_all())
    X, meta = featurise(sigs)
    f0 = np.array([m["f0_hz"] for m in meta])
    rows = []
    print("3rd harmonic stays under Nyquist while f0 < %.1f Hz (%.0f rpm)"
          % (MAX_F0_FOR_3X, MAX_F0_FOR_3X * 60))
    for tag, sel in [("all_speeds", np.ones(len(meta), bool)),
                     ("f0_below_40Hz_3x_intact", f0 < MAX_F0_FOR_3X),
                     ("f0_above_40Hz_3x_aliased", f0 >= MAX_F0_FOR_3X)]:
        if sel.sum() < 500:
            continue
        Xs = X[sel]
        ms = [m for m, s in zip(meta, sel) if s]
        base = evaluate(Xs, ms, tag, "all_dimensionless", rows)
        if base is None:
            continue
        print("\n%-26s n=%6d  all 13 features AUC=%.4f" % (tag, sel.sum(), base))
        for g in GROUPS:
            keep = [n for n in F.DIMENSIONLESS if n not in F.GROUPS[g]]
            v = evaluate(subset(Xs, keep), ms, tag, "drop_" + g, rows)
            if v is not None:
                print("   drop %-7s AUC=%.4f  delta=%+.4f" % (g, v, v - base))
    write_csv(table("t", "e9b_orders_by_speed.csv"), rows)
