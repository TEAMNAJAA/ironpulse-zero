import os, sys
import numpy as np
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)
sys.path.insert(0, os.path.dirname(_PROJ))
from experiments.common import CFG, cwru_index_48k, maf_index_all, write_csv, table
from core import pipeline, detectors, evaluate as EV

FS = float(CFG["camera"]["primary_fs_hz"])
EXP = CFG["camera"]["default_exposure_sec"]


def run(name, index):
    conds = [dict(fs_out=FS, mode="boxcar", exposure_sec=EXP, noise_type=None, snr_db=None)]
    for nt in CFG["noise"]["types"]:
        for snr in CFG["noise"]["snr_db"]:
            conds.append(dict(fs_out=FS, mode="boxcar", exposure_sec=EXP,
                              noise_type=nt, snr_db=snr))
    res = pipeline.build_multi(index, CFG, conds, domain="displacement", progress=400)
    out, per = [], []
    for ci, c in enumerate(conds):
        X, Xr, meta = res[ci]
        if not len(meta):
            continue
        rows, pl = EV.run_cv(X, meta, detectors.MahalanobisDetector, CFG)
        if not rows:
            continue
        a = EV.agg(rows)
        a.update(dataset=name, noise_type=c["noise_type"] or "none",
                 snr_db=c["snr_db"] if c["snr_db"] is not None else "",
                 fs_hz=FS, mode="boxcar")
        out.append(a)
        g = {}
        for p in pl:
            g.setdefault(p["label"], []).append(p)
        for lab, rs in sorted(g.items()):
            aa = EV.agg(rs)
            aa.update(dataset=name, noise_type=c["noise_type"] or "none",
                      snr_db=c["snr_db"] if c["snr_db"] is not None else "", label=lab)
            per.append(aa)
        print("  %-6s snr=%-4s AUC=%.4f +/- %.4f" % (c["noise_type"] or "none",
              c["snr_db"], a["roc_auc_mean"], a["roc_auc_std"]), flush=True)
    return out, per


if __name__ == "__main__":
    O, P = [], []
    for nm, idx in [("cwru_48k", cwru_index_48k()), ("mafaulda", maf_index_all())]:
        print("===", nm, len(idx), "files", flush=True)
        o, p = run(nm, idx)
        O += o
        P += p
    write_csv(table("t", "e6_noise_overall.csv"), O)
    write_csv(table("t", "e6_noise_per_label.csv"), P)
