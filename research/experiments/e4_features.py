import os, sys, copy
import numpy as np
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)
sys.path.insert(0, os.path.dirname(_PROJ))
from experiments.common import CFG, cwru_index_48k, maf_index_all, write_csv, table
from core import pipeline, detectors, evaluate as EV, features as F

FS = CFG["camera"]["primary_fs_hz"]
EXP = CFG["camera"]["default_exposure_sec"]


def subset(X, names):
    idx = [F.DIMENSIONLESS.index(n) for n in names]
    return X[:, idx]


def run(name, index):
    X, Xr, meta = pipeline.build(index, CFG, fs_out=float(FS), mode="boxcar",
                                 exposure_sec=EXP, domain="displacement", progress=300)
    if not len(meta):
        return []
    rows = []
    base = EV.agg(EV.run_cv(X, meta, detectors.MahalanobisDetector, CFG)[0])
    base.update(dataset=name, variant="all_dimensionless", dropped="", n_features=X.shape[1])
    rows.append(base)
    ref = base["roc_auc_mean"]
    for g, names in sorted(F.GROUPS.items()):
        keep = [n for n in F.DIMENSIONLESS if n not in names]
        if not keep:
            continue
        r = EV.agg(EV.run_cv(subset(X, keep), meta, detectors.MahalanobisDetector, CFG)[0])
        r.update(dataset=name, variant="drop_" + g, dropped=",".join(names),
                 n_features=len(keep), delta_auc=round(r["roc_auc_mean"] - ref, 4))
        rows.append(r)
        print("  drop %-8s AUC=%.4f (delta %+.4f)" % (g, r["roc_auc_mean"],
              r["roc_auc_mean"] - ref), flush=True)
    for g, names in sorted(F.GROUPS.items()):
        keep = [n for n in names if n in F.DIMENSIONLESS]
        r = EV.agg(EV.run_cv(subset(X, keep), meta, detectors.MahalanobisDetector, CFG)[0])
        r.update(dataset=name, variant="only_" + g, dropped="", n_features=len(keep),
                 delta_auc=round(r["roc_auc_mean"] - ref, 4))
        rows.append(r)
    rraw = EV.agg(EV.run_cv(Xr, meta, detectors.MahalanobisDetector, CFG)[0])
    rraw.update(dataset=name, variant="raw_units_only", dropped="",
                n_features=Xr.shape[1], delta_auc=round(rraw["roc_auc_mean"] - ref, 4))
    rows.append(rraw)
    both = np.hstack([X, Xr])
    rb = EV.agg(EV.run_cv(both, meta, detectors.MahalanobisDetector, CFG)[0])
    rb.update(dataset=name, variant="dimensionless_plus_raw", dropped="",
              n_features=both.shape[1], delta_auc=round(rb["roc_auc_mean"] - ref, 4))
    rows.append(rb)
    print("  all=%.4f  raw_only=%.4f  both=%.4f" % (ref, rraw["roc_auc_mean"],
          rb["roc_auc_mean"]), flush=True)
    return rows


if __name__ == "__main__":
    O = []
    for nm, idx in [("cwru_48k", cwru_index_48k()), ("mafaulda", maf_index_all())]:
        print("===", nm, len(idx), "files", flush=True)
        O += run(nm, idx)
    write_csv(table("t", "e4_features.csv"), O)
