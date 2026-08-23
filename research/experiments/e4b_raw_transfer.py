import os, sys
import numpy as np
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)
sys.path.insert(0, os.path.dirname(_PROJ))
from experiments.common import CFG, maf_index_all, cwru_index_48k, write_csv, table
from core import pipeline, detectors, evaluate as EV, features as F

FS = float(CFG["camera"]["primary_fs_hz"])
EXP = CFG["camera"]["default_exposure_sec"]
AMP = ["rms_px", "peak_px", "A1_px"]


def sets(X, Xr):
    amp = Xr[:, [F.RAW.index(k) for k in AMP]]
    logamp = np.log10(np.maximum(amp, 1e-12))
    return {"dimensionless": X,
            "dimensionless_plus_raw": np.hstack([X, logamp]),
            "raw_only": logamp}


def split_eval(X, meta, tr_mask, te_mask, tag, fsname, dataset):
    labs = np.array([m["label"] for m in meta])
    tr = tr_mask & (labs == "normal")
    if tr.sum() < 10 or te_mask.sum() == 0:
        return None
    det = detectors.MahalanobisDetector(CFG).fit(X[tr])
    thr = det.threshold()
    s = det.score(X[te_mask])
    y = (labs[te_mask] != "normal").astype(int)
    m = EV.binary_metrics(y, s, thr)
    m.update(dataset=dataset, featureset=fsname, split=tag, n_train=int(tr.sum()))
    return m


def run(dataset, index):
    X, Xr, meta = pipeline.build(index, CFG, fs_out=FS, mode="boxcar", exposure_sec=EXP,
                                 domain="displacement", progress=600)
    if not len(meta):
        return []
    f0 = np.array([m["f0_hz"] for m in meta])
    med = float(np.median(np.unique(f0)))
    lo, hi = f0 < med, f0 >= med
    rows = []
    for fsname, XX in sets(X, Xr).items():
        for tag, a, b in [("train_low_test_high", lo, hi),
                          ("train_high_test_low", hi, lo)]:
            r = split_eval(XX, meta, a, b, tag, fsname, dataset)
            if r is None:
                continue
            r["median_f0_hz"] = round(med, 3)
            rows.append(r)
            print("  %-24s %-20s AUC=%.4f  FAR=%.4f" % (fsname, tag, r["roc_auc"],
                  r["far"]), flush=True)
        rowsk, _ = EV.run_cv(XX, meta, detectors.MahalanobisDetector, CFG, per_label=False)
        if rowsk:
            a = EV.agg(rowsk)
            rows.append(dict(dataset=dataset, featureset=fsname, split="within_speed_5fold",
                             roc_auc=round(a["roc_auc_mean"], 4),
                             far=round(a["far_mean"], 4), n_train=0,
                             median_f0_hz=round(med, 3)))
            print("  %-24s %-20s AUC=%.4f  FAR=%.4f" % (fsname, "within_speed_5fold",
                  a["roc_auc_mean"], a["far_mean"]), flush=True)
    return rows


if __name__ == "__main__":
    O = []
    for nm, idx in [("mafaulda", maf_index_all()), ("cwru_48k", cwru_index_48k())]:
        print("===", nm, flush=True)
        O += run(nm, idx)
    write_csv(table("t", "e4b_raw_transfer.csv"), O)
