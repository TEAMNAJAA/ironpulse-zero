import os, sys
import numpy as np
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)
sys.path.insert(0, os.path.dirname(_PROJ))
from experiments.common import CFG, maf_index_all, cwru_index_48k, write_csv, table
from core import pipeline, detectors, evaluate as EV, camera_sim as cs

EXP = CFG["camera"]["default_exposure_sec"]
BANDS = [(10, 20), (20, 30), (30, 40), (40, 50), (50, 65)]


def run(name, index, fs_out, mode, exposure):
    X, Xr, meta = pipeline.build(index, CFG, fs_out=fs_out, mode=mode,
                                 exposure_sec=exposure, domain="displacement", progress=500)
    if not len(meta):
        return []
    fids = np.array([m["file_id"] for m in meta])
    labs = np.array([m["label"] for m in meta])
    f0 = np.array([m["f0_hz"] for m in meta])
    k, folds = EV.make_folds(fids, labs, CFG["evaluate"]["n_folds"], CFG["seed"])
    scores = np.full(len(meta), np.nan)
    for i in range(k):
        te = np.array([f in set(folds[i]) for f in fids])
        tr = (~te) & (labs == "normal")
        if tr.sum() < 5 or te.sum() == 0:
            continue
        det = detectors.MahalanobisDetector(CFG).fit(X[tr])
        scores[te] = det.score(X[te])
    ok = np.isfinite(scores)
    rows = []
    for lo, hi in BANDS:
        sel = ok & (f0 >= lo) & (f0 < hi)
        if sel.sum() < 30:
            continue
        y = (labs[sel] != "normal").astype(int)
        if len(set(y.tolist())) < 2:
            continue
        m = EV.binary_metrics(y, scores[sel], np.inf)
        med = float(np.median(f0[sel]))
        m.update(dataset=name, fs_hz=fs_out, mode=mode, band_lo=lo, band_hi=hi,
                 median_f0_hz=round(med, 2),
                 max_usable_order=round((fs_out / 2.0) / med, 2),
                 n_windows=int(sel.sum()))
        rows.append(m)
        print("  %-9s f0 %2d-%2d Hz  med=%5.1f  max_order=%5.2f  AUC=%.4f  (n=%d)" % (
            mode, lo, hi, med, m["max_usable_order"], m["roc_auc"], sel.sum()), flush=True)
    return rows


if __name__ == "__main__":
    O = []
    idx = maf_index_all()
    print("MAFAULDA files:", len(idx), flush=True)
    for fs_out, mode, exp in [(cs.BASE_FS, "ideal", None),
                              (480.0, "boxcar", EXP),
                              (240.0, "boxcar", EXP),
                              (120.0, "boxcar", EXP)]:
        print("--- fs=%g %s" % (fs_out, mode), flush=True)
        O += run("mafaulda", idx, fs_out, mode, exp)
    write_csv(table("t", "e2b_speed_band.csv"), O)
