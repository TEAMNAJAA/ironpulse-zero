import os, sys, time, json
import numpy as np
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)
sys.path.insert(0, os.path.dirname(_PROJ))
from experiments.common import CFG, cwru_index_48k, maf_index_all, write_csv, table
from core import pipeline, detectors, evaluate as EV, camera_sim as cs

RATES = CFG["camera"]["target_fs_hz"]
MODES = CFG["camera"]["decimation_modes"]
EXP = CFG["camera"]["default_exposure_sec"]


def conditions():
    cs_ = [dict(fs_out=cs.BASE_FS, mode="ideal", tag="full")]
    for m in MODES:
        for r in RATES:
            cs_.append(dict(fs_out=float(r), mode=m,
                            exposure_sec=EXP if m == "boxcar" else None, tag="sweep"))
    return cs_


def run(name, index, domain="displacement"):
    conds = conditions()
    t0 = time.time()
    res = pipeline.build_multi(index, CFG, conds, domain=domain, progress=200)
    print("%s %s: built %d conditions in %.0fs" % (name, domain, len(conds),
                                                   time.time() - t0), flush=True)
    overall, perlab = [], []
    for ci, c in enumerate(conds):
        X, Xr, meta = res[ci]
        if not len(meta):
            continue
        rows, per = EV.run_cv(X, meta, detectors.MahalanobisDetector, CFG)
        if not rows:
            continue
        a = EV.agg(rows)
        a.update(dataset=name, domain=domain, fs_hz=c["fs_out"], mode=c["mode"],
                 exposure_sec=c.get("exposure_sec"), n_windows=len(meta))
        overall.append(a)
        g = {}
        for p in per:
            g.setdefault(p["label"], []).append(p)
        for lab, rs in sorted(g.items()):
            aa = EV.agg(rs)
            aa.update(dataset=name, domain=domain, fs_hz=c["fs_out"], mode=c["mode"], label=lab)
            perlab.append(aa)
        print("  fs=%8.0f %-7s AUC=%.4f +/- %.4f" % (c["fs_out"], c["mode"],
              a["roc_auc_mean"], a["roc_auc_std"]), flush=True)
    return overall, perlab


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    O, P = [], []
    if which in ("both", "cwru"):
        o, p = run("cwru_48k", cwru_index_48k())
        O += o
        P += p
    if which in ("both", "mafaulda"):
        mi = maf_index_all()
        print("MAFAULDA files:", len(mi), flush=True)
        o, p = run("mafaulda", mi)
        O += o
        P += p
    suf = "" if which == "both" else "_" + which
    write_csv(table("t", "e2_sampling_rate_overall%s.csv" % suf), O)
    write_csv(table("t", "e2_sampling_rate_per_label%s.csv" % suf), P)
