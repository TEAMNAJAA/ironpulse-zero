import os, sys, copy, time
import numpy as np
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)
sys.path.insert(0, os.path.dirname(_PROJ))
from experiments.common import CFG, cwru_index_48k, maf_index_all, write_csv, table
from core import pipeline, camera_sim as cs, features as F, detectors, evaluate as EV
from core import signals as S

FS = float(CFG["camera"]["primary_fs_hz"])
EXP = CFG["camera"]["default_exposure_sec"]

RESEARCH_BANDS = {"E_sub": [0.10, 0.80], "E_1x": [0.80, 1.20],
                  "E_mid": [1.20, 5.00], "E_hi": [5.00, 40.00]}
APPSPEC_BANDS = {"E_sub": [0.20, 0.80], "E_1x": [0.80, 1.20],
                 "E_mid": [1.20, 3.50], "E_hi": [3.50, 40.00]}


def build_signals(index):
    out = []
    for i, rec in enumerate(index):
        sig, fsb, m = pipeline.signal_for(rec, CFG, domain="displacement")
        if sig is None:
            continue
        y = cs.decimate(sig, fsb, FS, mode="boxcar", exposure_sec=EXP)
        y = cs.to_pixels(y, CFG["optics"]["pixels_per_meter"])
        out.append((y, m["f0_hz"], m["label"], m["file_id"]))
        if i % 600 == 0:
            print("  signals %d/%d" % (i, len(index)), flush=True)
    return out


def featurise(sigs, cfg):
    X, meta = [], []
    for y, f0, lab, fid in sigs:
        for s0, w in S.windows(y, FS, cfg["signal"]["window_sec"], cfg["signal"]["hop_sec"]):
            f, raw, av = F.extract(w, FS, f0, cfg)
            X.append(F.vector(f))
            meta.append(dict(file_id=fid, label=lab, f0_hz=f0))
    return np.vstack(X), meta


def variant_cfg(**kw):
    c = copy.deepcopy(CFG)
    if "detrend" in kw:
        c["signal"]["detrend"] = kw["detrend"]
    if "window_fn" in kw:
        c["signal"]["window_fn"] = kw["window_fn"]
    if "tol" in kw:
        c["features"]["peak_search_halfwidth_order"] = kw["tol"]
    if "bands" in kw:
        c["features"]["bands"] = copy.deepcopy(kw["bands"])
    if "order_floor" in kw:
        c["integration"]["highpass_order_floor"] = kw["order_floor"]
    return c


def score_variant(sigs, cfg, name, group, dataset, det="mahalanobis"):
    X, meta = featurise(sigs, cfg)
    rows, _ = EV.run_cv(X, meta, detectors.BY_NAME[det], cfg, per_label=False)
    if not rows:
        return None
    a = EV.agg(rows)
    a.update(dataset=dataset, group=group, variant=name, detector=det,
             n_windows=len(meta), n_features=X.shape[1])
    print("  %-10s %-26s AUC=%.4f +/- %.4f  F1=%.4f  FAR=%.4f" % (
        group, name, a["roc_auc_mean"], a["roc_auc_std"], a["f1_mean"], a["far_mean"]),
        flush=True)
    return a


def threshold_sweep(sigs, cfg, dataset):
    X, meta = featurise(sigs, cfg)
    fids = np.array([m["file_id"] for m in meta])
    labs = np.array([m["label"] for m in meta])
    k, folds = EV.make_folds(fids, labs, CFG["evaluate"]["n_folds"], CFG["seed"])
    out = []
    for pct in [90.0, 95.0, 97.5, 99.0, 99.5]:
        c = copy.deepcopy(cfg)
        c["detectors"]["threshold_percentile"] = pct
        f1s, fars, aucs = [], [], []
        for i in range(k):
            te = np.array([f in set(folds[i]) for f in fids])
            tr = (~te) & (labs == "normal")
            if tr.sum() < 5 or te.sum() == 0:
                continue
            det = detectors.MahalanobisDetector(c).fit(X[tr])
            thr = det.threshold()
            s = det.score(X[te])
            y = (labs[te] != "normal").astype(int)
            m = EV.binary_metrics(y, s, thr)
            f1s.append(m["f1"])
            fars.append(m["far"])
            aucs.append(m["roc_auc"])
        if not f1s:
            continue
        r = dict(dataset=dataset, group="threshold_percentile", variant="%g" % pct,
                 roc_auc_mean=round(float(np.mean(aucs)), 4),
                 f1_mean=round(float(np.mean(f1s)), 4),
                 far_mean=round(float(np.mean(fars)), 4),
                 f1_minus_far=round(float(np.mean(f1s) - np.mean(fars)), 4))
        out.append(r)
        print("  %-10s pct=%-5s  AUC=%.4f  F1=%.4f  FAR=%.4f" % (
            "threshold", "%g" % pct, r["roc_auc_mean"], r["f1_mean"], r["far_mean"]), flush=True)
    return out


def run(dataset, index):
    t0 = time.time()
    sigs = build_signals(index)
    print("%s: %d signals in %.0fs" % (dataset, len(sigs), time.time() - t0), flush=True)
    rows = []
    for d in ["constant", "linear", "poly2", "poly3"]:
        r = score_variant(sigs, variant_cfg(detrend=d), d, "detrend", dataset)
        if r:
            rows.append(r)
    for w in ["hann", "hamming", "none"]:
        r = score_variant(sigs, variant_cfg(window_fn=w), w, "window_fn", dataset)
        if r:
            rows.append(r)
    for t in [0.02, 0.05, 0.08, 0.10, 0.15, 0.20]:
        r = score_variant(sigs, variant_cfg(tol=t), "%g" % t, "order_tolerance", dataset)
        if r:
            rows.append(r)
    for nm, b in [("research_bands", RESEARCH_BANDS), ("appspec_bands", APPSPEC_BANDS)]:
        r = score_variant(sigs, variant_cfg(bands=b), nm, "bands", dataset)
        if r:
            rows.append(r)
    for fl in [0.10, 0.20]:
        r = score_variant(sigs, variant_cfg(order_floor=fl), "%g" % fl, "rms_order_min", dataset)
        if r:
            rows.append(r)
    rows.extend(threshold_sweep(sigs, CFG, dataset))
    return rows


if __name__ == "__main__":
    O = []
    for nm, idx in [("mafaulda", maf_index_all()), ("cwru_48k", cwru_index_48k())]:
        print("===", nm, len(idx), "files", flush=True)
        O += run(nm, idx)
    write_csv(table("t", "e9_dsp_params.csv"), O)
