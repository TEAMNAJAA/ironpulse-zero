import os, sys
import numpy as np
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)
sys.path.insert(0, os.path.dirname(_PROJ))
from experiments.common import CFG, cwru_index_48k, maf_index_all, write_csv, table
from core import pipeline, detectors, evaluate as EV

RATES = [480.0, 240.0, 120.0]
EXPOSURES = CFG["camera"]["exposure_sec"]


def run(name, index):
    conds, tags = [], []
    for fs in RATES:
        conds.append(dict(fs_out=fs, mode="ideal"))
        tags.append((fs, "ideal", None))
        for e in EXPOSURES:
            conds.append(dict(fs_out=fs, mode="boxcar", exposure_sec=e))
            tags.append((fs, "boxcar", e))
    res = pipeline.build_multi(index, CFG, conds, domain="displacement", progress=500)
    out = []
    for ci, (fs, mode, e) in enumerate(tags):
        X, Xr, meta = res[ci]
        if not len(meta):
            continue
        rows, _ = EV.run_cv(X, meta, detectors.MahalanobisDetector, CFG, per_label=False)
        if not rows:
            continue
        a = EV.agg(rows)
        shutter = ("1/%d s" % round(1.0 / e)) if e else "none"
        duty = (e * fs) if e else 0.0
        a.update(dataset=name, fs_hz=fs, mode=mode,
                 exposure_sec=e if e else "", shutter=shutter,
                 exposure_duty=round(min(duty, 1.0), 4), n_windows=len(meta))
        out.append(a)
        print("  fs=%5.0f %-7s shutter=%-8s duty=%.2f  AUC=%.4f +/- %.4f" % (
            fs, mode, shutter, min(duty, 1.0), a["roc_auc_mean"], a["roc_auc_std"]), flush=True)
    return out


if __name__ == "__main__":
    O = []
    for nm, idx in [("mafaulda", maf_index_all()), ("cwru_48k", cwru_index_48k())]:
        print("===", nm, len(idx), "files", flush=True)
        O += run(nm, idx)
    write_csv(table("t", "e2c_exposure.csv"), O)
