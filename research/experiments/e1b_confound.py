import os, sys
import numpy as np
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)
sys.path.insert(0, os.path.dirname(_PROJ))
from experiments.common import CFG, maf_index_all, write_csv, table
from core import pipeline, detectors, evaluate as EV, camera_sim as cs

EXP = CFG["camera"]["default_exposure_sec"]
BEARING = ("cage", "outer_race", "ball")


def cv_scores(X, meta):
    fids = np.array([m["file_id"] for m in meta])
    labs = np.array([m["label"] for m in meta])
    k, folds = EV.make_folds(fids, labs, CFG["evaluate"]["n_folds"], CFG["seed"])
    s = np.full(len(meta), np.nan)
    thr = []
    for i in range(k):
        te = np.array([f in set(folds[i]) for f in fids])
        tr = (~te) & (labs == "normal")
        if tr.sum() < 5 or te.sum() == 0:
            continue
        det = detectors.MahalanobisDetector(CFG).fit(X[tr])
        s[te] = det.score(X[te])
        thr.append(det.threshold())
    return s, (float(np.mean(thr)) if thr else np.nan)


def run(tag, fs_out, mode, exposure):
    X, Xr, meta = pipeline.build(maf_index_all(), CFG, fs_out=fs_out, mode=mode,
                                 exposure_sec=exposure, domain="displacement", progress=600)
    if not len(meta):
        return []
    labs = np.array([m["label"] for m in meta])
    sev = np.array([m["severity"] if m["severity"] is not None else -1.0 for m in meta])
    s, thr = cv_scores(X, meta)
    ok = np.isfinite(s)
    base = ok & (labs == "normal")
    rows = []
    for lab in sorted(set(labs.tolist())):
        if lab == "normal":
            continue
        is_bearing = any(b in lab for b in BEARING)
        groups = [("all_severities", ok & (labs == lab))]
        if is_bearing:
            groups.append(("added_mass_0g_only", ok & (labs == lab) & (sev == 0.0)))
            groups.append(("added_mass_above_0g", ok & (labs == lab) & (sev > 0.0)))
        for gname, sel in groups:
            if sel.sum() < 30 or base.sum() < 30:
                continue
            y = np.r_[np.zeros(int(base.sum())), np.ones(int(sel.sum()))]
            ss = np.r_[s[base], s[sel]]
            m = EV.binary_metrics(y, ss, thr)
            m.update(condition=tag, fs_hz=fs_out, mode=mode, label=lab, subset=gname,
                     n_fault_windows=int(sel.sum()))
            rows.append(m)
    return rows


if __name__ == "__main__":
    O = []
    for tag, fs_out, mode, exp in [("full_48kHz", cs.BASE_FS, "ideal", None),
                                   ("240Hz_boxcar", 240.0, "boxcar", EXP)]:
        print("===", tag, flush=True)
        rr = run(tag, fs_out, mode, exp)
        O += rr
        for r in rr:
            print("  %-24s %-20s AUC=%.4f  (n=%d)" % (r["label"], r["subset"],
                  r["roc_auc"], r["n_fault_windows"]), flush=True)
    write_csv(table("t", "e1b_confound.csv"), O)
