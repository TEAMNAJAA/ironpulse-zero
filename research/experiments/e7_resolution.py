import os, sys
import numpy as np
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)
sys.path.insert(0, os.path.dirname(_PROJ))
from experiments.common import CFG, cwru_index_48k, maf_index_all, write_csv, table
from core import pipeline, detectors, evaluate as EV

FS = float(CFG["camera"]["primary_fs_hz"])
EXP = CFG["camera"]["default_exposure_sec"]
PPM = CFG["optics"]["pixels_per_meter"]


def run(name, index):
    qs = [None] + list(CFG["camera"]["quantization_px"])
    conds = [dict(fs_out=FS, mode="boxcar", exposure_sec=EXP, quant_px=q) for q in qs]
    res = pipeline.build_multi(index, CFG, conds, domain="displacement", progress=400)
    out, per = [], []
    for ci, c in enumerate(conds):
        X, Xr, meta = res[ci]
        if not len(meta):
            continue
        rms = Xr[:, 0]
        dead = float(np.mean(rms <= 0.0))
        rows, pl = EV.run_cv(X, meta, detectors.MahalanobisDetector, CFG)
        if not rows:
            continue
        a = EV.agg(rows)
        q = c["quant_px"]
        a.update(dataset=name, quant_px=q if q else 0.0,
                 quant_um=round(q / PPM * 1e6, 3) if q else 0.0,
                 dead_window_frac=round(dead, 4),
                 median_rms_px=round(float(np.median(rms)), 6), fs_hz=FS, mode="boxcar")
        out.append(a)
        g = {}
        for p in pl:
            g.setdefault(p["label"], []).append(p)
        for lab, rs in sorted(g.items()):
            aa = EV.agg(rs)
            aa.update(dataset=name, quant_px=q if q else 0.0, label=lab)
            per.append(aa)
        print("  quant=%-7s (%7.2f um) AUC=%.4f +/- %.4f  dead=%.3f  med_rms=%.5f px" % (
            q, a["quant_um"], a["roc_auc_mean"], a["roc_auc_std"], dead,
            a["median_rms_px"]), flush=True)
    return out, per


if __name__ == "__main__":
    O, P = [], []
    for nm, idx in [("cwru_48k", cwru_index_48k()), ("mafaulda", maf_index_all())]:
        print("===", nm, len(idx), "files", flush=True)
        o, p = run(nm, idx)
        O += o
        P += p
    write_csv(table("t", "e7_resolution_overall.csv"), O)
    write_csv(table("t", "e7_resolution_per_label.csv"), P)
