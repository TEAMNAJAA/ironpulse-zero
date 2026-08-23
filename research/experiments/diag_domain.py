import os, sys, copy
import numpy as np
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)
sys.path.insert(0, os.path.dirname(_PROJ))
from experiments.common import CFG, cwru_index_48k, write_csv, table
from core import pipeline, detectors, evaluate as EV, camera_sim as cs

idx = cwru_index_48k()
print("files:", len(idx))
rows = []
for domain in ["acceleration", "displacement"]:
    for mo in [40.0, 200.0, 800.0]:
        cfg = copy.deepcopy(CFG)
        cfg["features"]["max_order"] = mo
        cfg["features"]["bands"]["E_hi"] = [5.0, mo]
        X, Xr, meta = pipeline.build(idx, cfg, fs_out=cs.BASE_FS, mode="ideal", domain=domain)
        r, per = EV.run_cv(X, meta, detectors.MahalanobisDetector, cfg)
        a = EV.agg(r)
        a.update(domain=domain, max_order=mo, n_windows=len(meta))
        pl = {}
        for p in per:
            pl.setdefault(p["label"], []).append(p["roc_auc"])
        for k, v in pl.items():
            a["auc_" + k] = round(float(np.nanmean(v)), 4)
        rows.append(a)
        print("%-13s max_order=%5.0f  AUC=%.4f +/- %.4f   %s" % (
            domain, mo, a["roc_auc_mean"], a["roc_auc_std"],
            {k: a["auc_" + k] for k in pl}), flush=True)
write_csv(table("t", "diag_domain_cwru.csv"), rows)
