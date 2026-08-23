import os, sys, time
import numpy as np
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)
sys.path.insert(0, os.path.dirname(_PROJ))
from experiments.common import CFG, cwru_index_48k, maf_index_all, write_csv, table
from core import pipeline, detectors, evaluate as EV, camera_sim as cs

RATES = CFG["camera"]["target_fs_hz"]
EXP = CFG["camera"]["default_exposure_sec"]
DETECTORS = ["mahalanobis", "ocsvm"]


def conditions():
    out = [dict(fs_out=cs.BASE_FS, mode="ideal")]
    for r in RATES:
        out.append(dict(fs_out=float(r), mode="boxcar", exposure_sec=EXP))
    return out


def run(name, index):
    conds = conditions()
    t0 = time.time()
    res = pipeline.build_multi(index, CFG, conds, domain="displacement", progress=600)
    print("%s: built %d conditions in %.0fs" % (name, len(conds), time.time() - t0), flush=True)
    overall, perlab = [], []
    for ci, c in enumerate(conds):
        X, Xr, meta = res[ci]
        if not len(meta):
            continue
        for dname in DETECTORS:
            cls = detectors.BY_NAME[dname]
            rows, per = EV.run_cv(X, meta, cls, CFG)
            if not rows:
                continue
            a = EV.agg(rows)
            a.update(dataset=name, detector=dname, fs_hz=c["fs_out"], mode=c["mode"],
                     n_windows=len(meta))
            overall.append(a)
            g = {}
            for p in per:
                g.setdefault(p["label"], []).append(p)
            for lab, rs in sorted(g.items()):
                aa = EV.agg(rs)
                aa.update(dataset=name, detector=dname, fs_hz=c["fs_out"], label=lab)
                perlab.append(aa)
        m = [r for r in overall if r["fs_hz"] == c["fs_out"] and r["dataset"] == name]
        got = {r["detector"]: r["roc_auc_mean"] for r in m}
        print("  fs=%8.0f  %s" % (c["fs_out"],
              "  ".join("%s=%.4f" % (k, v) for k, v in sorted(got.items()))), flush=True)
    return overall, perlab


if __name__ == "__main__":
    O, P = [], []
    for nm, idx in [("mafaulda", maf_index_all()), ("cwru_48k", cwru_index_48k())]:
        print("===", nm, len(idx), "files", flush=True)
        o, p = run(nm, idx)
        O += o
        P += p
    write_csv(table("t", "e2d_best_detector_overall.csv"), O)
    write_csv(table("t", "e2d_best_detector_per_label.csv"), P)
