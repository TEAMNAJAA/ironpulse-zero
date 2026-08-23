import os, sys
import numpy as np
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)
sys.path.insert(0, os.path.dirname(_PROJ))
from experiments.common import CFG, cwru_index_48k, maf_index_all, write_csv, table
from core import pipeline, detectors, evaluate as EV

FS = float(CFG["camera"]["primary_fs_hz"])
EXP = CFG["camera"]["default_exposure_sec"]
F0_REF = 30.0


def split_eval(X, meta, train_mask, test_mask, tag, axis, dataset):
    labs = np.array([m["label"] for m in meta])
    tr = train_mask & (labs == "normal")
    if tr.sum() < 10 or test_mask.sum() == 0:
        return None
    det = detectors.MahalanobisDetector(CFG).fit(X[tr])
    thr = det.threshold()
    s = det.score(X[test_mask])
    y = (labs[test_mask] != "normal").astype(int)
    m = EV.binary_metrics(y, s, thr)
    m.update(dataset=dataset, axis=axis, split=tag, n_train=int(tr.sum()))
    return m


def speed_transfer(index, dataset):
    rows = []
    for axis, ovr in [("order", None), ("hz_fixed", F0_REF)]:
        X, Xr, meta = pipeline.build(index, CFG, fs_out=FS, mode="boxcar", exposure_sec=EXP,
                                     domain="displacement", f0_override=ovr, progress=400)
        if not len(meta):
            continue
        f0 = np.array([m["f0_hz"] for m in meta])
        med = float(np.median(np.unique(f0)))
        lo = f0 < med
        hi = f0 >= med
        allm = np.ones(len(meta), bool)
        for tag, a, b in [("train_low_test_high", lo, hi),
                          ("train_high_test_low", hi, lo),
                          ("same_speed_reference", allm, allm)]:
            r = split_eval(X, meta, a, b, tag, axis, dataset)
            if r is None:
                continue
            r["median_f0_hz"] = round(med, 3)
            rows.append(r)
            print("  %-9s %-22s AUC=%.4f  PR=%.4f  FAR=%.4f" % (
                axis, tag, r["roc_auc"], r["pr_auc"], r["far"]), flush=True)
    return rows


def machine_transfer():
    rows = []
    for axis, ovr in [("order", None), ("hz_fixed", F0_REF)]:
        Xa, _, ma = pipeline.build(maf_index_all(), CFG, fs_out=FS, mode="boxcar",
                                   exposure_sec=EXP, domain="displacement", f0_override=ovr)
        Xb, _, mb = pipeline.build(cwru_index_48k(), CFG, fs_out=FS, mode="boxcar",
                                   exposure_sec=EXP, domain="displacement", f0_override=ovr)
        if not len(ma) or not len(mb):
            continue
        la = np.array([m["label"] for m in ma])
        lb = np.array([m["label"] for m in mb])
        det = detectors.MahalanobisDetector(CFG).fit(Xa[la == "normal"])
        thr = det.threshold()
        s = det.score(Xb)
        y = (lb != "normal").astype(int)
        r = EV.binary_metrics(y, s, thr)
        r.update(dataset="train_mafaulda_test_cwru", axis=axis, split="cross_machine",
                 n_train=int((la == "normal").sum()))
        rows.append(r)
        print("  cross-machine %-9s AUC=%.4f  FAR=%.4f" % (axis, r["roc_auc"], r["far"]),
              flush=True)
    return rows


if __name__ == "__main__":
    O = []
    print("=== MAFAULDA cross-speed", flush=True)
    O += speed_transfer(maf_index_all(), "mafaulda")
    print("=== CWRU cross-load", flush=True)
    O += speed_transfer(cwru_index_48k(), "cwru_48k")
    print("=== cross-machine", flush=True)
    O += machine_transfer()
    write_csv(table("t", "e5_transfer.csv"), O)
