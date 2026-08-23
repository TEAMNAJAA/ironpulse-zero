import os, sys, time
import numpy as np
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)
sys.path.insert(0, os.path.dirname(_PROJ))
from experiments.common import CFG, cwru_index_48k, maf_index_all, write_csv, table
from core import pipeline, detectors, evaluate as EV, camera_sim as cs

FS = CFG["camera"]["primary_fs_hz"]
EXP = CFG["camera"]["default_exposure_sec"]


def run(name, index):
    out, per = [], []
    X, Xr, meta = pipeline.build(index, CFG, fs_out=float(FS), mode="boxcar",
                                 exposure_sec=EXP, domain="displacement", progress=300)
    if not len(meta):
        return out, per
    for cls in detectors.ALL:
        t0 = time.time()
        rows, pl = EV.run_cv(X, meta, cls, CFG)
        if not rows:
            continue
        a = EV.agg(rows)
        a.update(dataset=name, detector=cls.name, fs_hz=FS, mode="boxcar",
                 n_windows=len(meta), fit_score_s=round(time.time() - t0, 2))
        out.append(a)
        g = {}
        for p in pl:
            g.setdefault(p["label"], []).append(p)
        for lab, rs in sorted(g.items()):
            aa = EV.agg(rs)
            aa.update(dataset=name, detector=cls.name, label=lab)
            per.append(aa)
        print("  %-18s AUC=%.4f +/- %.4f  PR=%.4f  F1=%.4f  FAR=%.4f  (%.1fs)" % (
            cls.name, a["roc_auc_mean"], a["roc_auc_std"], a["pr_auc_mean"],
            a["f1_mean"], a["far_mean"], a["fit_score_s"]), flush=True)
    return out, per


if __name__ == "__main__":
    O, P = [], []
    for nm, idx in [("cwru_48k", cwru_index_48k()), ("mafaulda", maf_index_all())]:
        print("===", nm, len(idx), "files", flush=True)
        o, p = run(nm, idx)
        O += o
        P += p
    write_csv(table("t", "e3_detectors_overall.csv"), O)
    write_csv(table("t", "e3_detectors_per_label.csv"), P)
