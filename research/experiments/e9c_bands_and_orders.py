import os, sys, copy
import numpy as np
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)
sys.path.insert(0, os.path.dirname(_PROJ))
from experiments.common import CFG, maf_index_all, cwru_index_48k, write_csv, table
from core import pipeline, camera_sim as cs, features as F, detectors, evaluate as EV
from core import signals as S

FS = float(CFG["camera"]["primary_fs_hz"])
EXP = CFG["camera"]["default_exposure_sec"]
NYQ_ORDER_LIMIT = (FS / 2.0) / 3.0


def build(index):
    out = []
    for rec in index:
        sig, fsb, m = pipeline.signal_for(rec, CFG, domain="displacement")
        if sig is None:
            continue
        y = cs.decimate(sig, fsb, FS, mode="boxcar", exposure_sec=EXP)
        y = cs.to_pixels(y, CFG["optics"]["pixels_per_meter"])
        out.append((y, m["f0_hz"], m["label"], m["file_id"]))
    return out


def featurise(sigs, cfg):
    X, meta, hi_empty = [], [], 0
    for y, f0, lab, fid in sigs:
        nyq_order = (FS / 2.0) / f0
        for s0, w in S.windows(y, FS, cfg["signal"]["window_sec"], cfg["signal"]["hop_sec"]):
            f, raw, av = F.extract(w, FS, f0, cfg)
            X.append(F.vector(f))
            meta.append(dict(file_id=fid, label=lab, f0_hz=f0))
            if not av.get("E_hi", False) or nyq_order <= cfg["features"]["bands"]["E_hi"][0]:
                hi_empty += 1
    return np.vstack(X), meta, hi_empty / max(len(meta), 1)


def cfg_bands(sub_lo, mid_hi):
    c = copy.deepcopy(CFG)
    c["features"]["bands"] = {"E_sub": [sub_lo, 0.80], "E_1x": [0.80, 1.20],
                              "E_mid": [1.20, mid_hi], "E_hi": [mid_hi, 40.00]}
    return c


def ev(X, meta, cfg, dataset, group, variant, extra=None):
    rows_, _ = EV.run_cv(X, meta, detectors.MahalanobisDetector, cfg, per_label=False)
    if not rows_:
        return None
    a = EV.agg(rows_)
    a.update(dataset=dataset, group=group, variant=variant, n_windows=len(meta))
    if extra:
        a.update(extra)
    return a


def subset(X, names):
    return X[:, [F.DIMENSIONLESS.index(n) for n in names]]


if __name__ == "__main__":
    rows = []
    for dataset, index in [("mafaulda", maf_index_all()), ("cwru_48k", cwru_index_48k())]:
        print("===", dataset, flush=True)
        sigs = build(index)
        print("  %d signals" % len(sigs), flush=True)
        print("  -- band decomposition (sub lower edge x mid/hi boundary)", flush=True)
        for sub_lo in [0.10, 0.20]:
            for mid_hi in [3.50, 5.00]:
                c = cfg_bands(sub_lo, mid_hi)
                X, meta, frac = featurise(sigs, c)
                a = ev(X, meta, c, dataset, "bands",
                       "sub%.2f_midhi%.2f" % (sub_lo, mid_hi),
                       dict(E_hi_empty_frac=round(frac, 4)))
                if a:
                    rows.append(a)
                    print("     sub_lo=%.2f mid/hi=%.2f  AUC=%.4f +/- %.4f  E_hi empty in %.1f%% of windows"
                          % (sub_lo, mid_hi, a["roc_auc_mean"], a["roc_auc_std"], 100 * frac),
                          flush=True)
        print("  -- order-group ablation split by whether 3x survives Nyquist", flush=True)
        X, meta, _ = featurise(sigs, CFG)
        f0 = np.array([m["f0_hz"] for m in meta])
        for tag, sel in [("all_speeds", np.ones(len(meta), bool)),
                         ("f0_under_40Hz_3x_intact", f0 < NYQ_ORDER_LIMIT),
                         ("f0_over_40Hz_3x_aliased", f0 >= NYQ_ORDER_LIMIT)]:
            if sel.sum() < 400:
                continue
            Xs = X[sel]
            ms = [m for m, s in zip(meta, sel) if s]
            base = ev(Xs, ms, CFG, dataset, "orders_" + tag, "all_dimensionless")
            if base is None:
                continue
            rows.append(base)
            keep = [n for n in F.DIMENSIONLESS if n not in F.GROUPS["orders"]]
            d = ev(subset(Xs, keep), ms, CFG, dataset, "orders_" + tag, "drop_orders")
            if d:
                rows.append(d)
                print("     %-26s n=%6d  all=%.4f  drop_orders=%.4f  delta=%+.4f"
                      % (tag, sel.sum(), base["roc_auc_mean"], d["roc_auc_mean"],
                         d["roc_auc_mean"] - base["roc_auc_mean"]), flush=True)
    write_csv(table("t", "e9c_bands_and_orders.csv"), rows)
