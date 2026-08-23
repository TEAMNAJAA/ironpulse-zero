import os, sys, time
import numpy as np
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)
sys.path.insert(0, os.path.dirname(_PROJ))
from experiments.common import CFG, cwru_index_48k, maf_index_all, write_csv, table
from core import pipeline, detectors, evaluate as EV, camera_sim as cs

GATE = 0.85


def run(name, index, cfg, domain, det_name="mahalanobis"):
    t0 = time.time()
    X, Xr, meta = pipeline.build(index, cfg, fs_out=cs.BASE_FS, mode="ideal",
                                 domain=domain, progress=200)
    if not len(meta):
        return None, []
    cls = detectors.BY_NAME[det_name]
    rows, per = EV.run_cv(X, meta, cls, cfg)
    a = EV.agg(rows)
    a.update(dataset=name, domain=domain, detector=det_name, fs_hz=cs.BASE_FS,
             mode="ideal", n_windows=len(meta),
             n_files=len(set(m["file_id"] for m in meta)),
             elapsed_s=round(time.time() - t0, 1))
    g = {}
    for p in per:
        g.setdefault(p["label"], []).append(p)
    perrows = []
    for lab, rs in sorted(g.items()):
        aa = EV.agg(rs)
        aa.update(dataset=name, domain=domain, label=lab, detector=det_name)
        perrows.append(aa)
    return a, perrows


if __name__ == "__main__":
    overall, per = [], []
    for name, idx in [("cwru_48k", cwru_index_48k()), ("mafaulda", maf_index_all())]:
        print("=== %s : %d files" % (name, len(idx)), flush=True)
        for domain in ["acceleration", "displacement"]:
            a, p = run(name, idx, CFG, domain)
            if not a:
                continue
            overall.append(a)
            per.extend(p)
            print("  %-13s AUC=%.4f +/- %.4f  PR=%.4f  F1=%.4f  FAR=%.4f  (%d win, %.0fs)" % (
                domain, a["roc_auc_mean"], a["roc_auc_std"], a["pr_auc_mean"],
                a["f1_mean"], a["far_mean"], a["n_windows"], a["elapsed_s"]), flush=True)
    write_csv(table("t", "e1_baseline_overall.csv"), overall)
    write_csv(table("t", "e1_baseline_per_label.csv"), per)
    gate = [a for a in overall if a["dataset"] == "cwru_48k" and a["domain"] == "acceleration"]
    if gate:
        v = gate[0]["roc_auc_mean"]
        print("\nE1 GATE (CWRU acceleration, full rate): AUC=%.4f  -> %s" % (
            v, "PASS" if v >= GATE else "FAIL"))
