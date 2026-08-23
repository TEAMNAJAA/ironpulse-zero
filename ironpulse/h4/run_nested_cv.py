import os
import sys
import io
import csv
import json
import yaml
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
REPO = os.path.dirname(APP)
sys.path.insert(0, REPO)
sys.path.insert(0, APP)
os.environ.setdefault("IRONPULSE_PROJECT_ROOT", APP)

from core import detectors
from appcore import cfgmap

NPZ = os.path.join(HERE, "features.npz")
RES = os.path.join(HERE, "results")
TABLES = os.path.join(RES, "tables")
SEEDS = [42, 43, 44, 45, 46]
MODELS = ["mahalanobis", "isolation_forest", "ocsvm", "pca_recon"]
PCA_MODELS = ("mahalanobis", "pca_recon")
PCA_SETTINGS = [3, 5, 8, "auto"]
FEATURE_SETS = ["all", "machine", "primary", "orderband"]
ORDER_BAND = ("r05", "r1", "r2", "r3", "rbp", "E_sub", "E_1x", "E_mid", "E_hi")
LINE = "=" * 100


def core_cfg_for(app_cfg, pca_setting):
    c = cfgmap.core_cfg(app_cfg, 240.0, 19.0, None)
    if pca_setting != "auto":
        c["detectors"]["pca_variance"] = int(pca_setting)
        c["detectors"]["pca_reconstruction"]["variance"] = int(pca_setting)
    return c


def columns(names, fs_name, primary_axis):
    if fs_name == "all":
        return np.arange(len(names))
    if fs_name == "machine":
        return np.array([i for i, n in enumerate(names) if n.startswith("d_machine_")])
    if fs_name == "primary":
        return np.array([i for i, n in enumerate(names) if n.startswith(primary_axis + ".")])
    if fs_name == "orderband":
        return np.array([i for i, n in enumerate(names)
                         if n.split(".", 1)[1] in ORDER_BAND])
    raise ValueError(fs_name)


def valid_combo(model, pca_setting, n_train, n_cols):
    if model not in PCA_MODELS:
        return pca_setting == "auto"
    if pca_setting == "auto":
        return True
    return int(pca_setting) <= min(n_train - 1, n_cols)


def fit_score(model, ccfg, seed, Xtr, Xte):
    det = detectors.BY_NAME[model](ccfg, seed=seed).fit(Xtr)
    return det, det.score(Xte)


def tightness(train_scores, held_scores):
    ref = float(np.percentile(train_scores, 95))
    spread = ref - float(np.percentile(train_scores, 50))
    if not np.isfinite(ref) or spread <= 0:
        return float("inf")
    return float((np.percentile(held_scores, 95) - np.median(train_scores)) / spread)


def inner_select(app_cfg, Xn, sess_n, seed, primary_axis, names, log):
    inner_sessions = sorted(set(sess_n))
    best = None
    for model in MODELS:
        for pca_setting in PCA_SETTINGS:
            for fs_name in FEATURE_SETS:
                cols = columns(names, fs_name, primary_axis)
                min_train = min(sum(1 for v in sess_n if v != s) for s in inner_sessions)
                if not valid_combo(model, pca_setting, min_train, len(cols)):
                    continue
                ccfg = core_cfg_for(app_cfg, pca_setting)
                per_fold = []
                ok = True
                for s in inner_sessions:
                    tr = np.array([i for i, v in enumerate(sess_n) if v != s])
                    te = np.array([i for i, v in enumerate(sess_n) if v == s])
                    if len(tr) < 4:
                        ok = False
                        break
                    try:
                        det, sc = fit_score(model, ccfg, seed, Xn[np.ix_(tr, cols)],
                                            Xn[np.ix_(te, cols)])
                    except Exception:
                        ok = False
                        break
                    if not np.all(np.isfinite(sc)):
                        ok = False
                        break
                    per_fold.append(tightness(det.train_scores, sc))
                if not ok or not per_fold or not np.all(np.isfinite(per_fold)):
                    continue
                crit = float(np.mean(per_fold))
                log.append(dict(model=model, pca=str(pca_setting), features=fs_name,
                                criterion=round(crit, 6), n_cols=len(cols)))
                if best is None or crit < best[0]:
                    best = (crit, model, pca_setting, fs_name)
    return best


def run(app_cfg, X, names, clips, groups, sessions, primary, level, out_rows,
        sel_rows, cand_rows):
    is_normal = np.array([g == "normal" for g in groups])
    normal_idx = np.nonzero(is_normal)[0]
    fault_idx = np.nonzero(~is_normal)[0]
    outer_sessions = sorted(set(sessions[normal_idx]))
    print("%s level: %d normal in %d sessions, %d fault"
          % (level, len(normal_idx), len(outer_sessions), len(fault_idx)))
    for seed in SEEDS:
        for fold, s in enumerate(outer_sessions):
            tr = normal_idx[sessions[normal_idx] != s]
            te = normal_idx[sessions[normal_idx] == s]
            vals, counts = np.unique(primary[tr], return_counts=True)
            primary_axis = str(vals[int(np.argmax(counts))])
            log = []
            best = inner_select(app_cfg, X[tr], sessions[tr], seed, primary_axis,
                                names, log)
            if best is None:
                print("  seed %d fold %s: no valid candidate" % (seed, s))
                continue
            crit, model, pca_setting, fs_name = best
            cols = columns(names, fs_name, primary_axis)
            ccfg = core_cfg_for(app_cfg, pca_setting)
            det = detectors.BY_NAME[model](ccfg, seed=seed).fit(X[np.ix_(tr, cols)])
            thr = det.threshold() * float(app_cfg["model"]["threshold_margin"])
            if not np.isfinite(thr) or thr <= 0:
                thr = float(np.max(det.train_scores)) or 1.0
            for i in np.concatenate([te, fault_idx]):
                sc = float(det.score(X[np.ix_([i], cols)])[0])
                out_rows.append(dict(level=level, seed=seed, fold=s, model=model,
                                     pca=str(pca_setting), features=fs_name,
                                     clip=clips[i], group=groups[i],
                                     session=sessions[i],
                                     is_fault=int(groups[i] != "normal"),
                                     score=round(sc, 8), threshold=round(thr, 8),
                                     norm_score=round(sc / thr, 8),
                                     flagged=int(sc > thr)))
            for c in log:
                c2 = dict(c)
                c2.update(level=level, seed=seed, fold=s)
                cand_rows.append(c2)
            sel_rows.append(dict(level=level, seed=seed, fold=s, n_train=len(tr),
                                 n_test_normal=len(te), model=model,
                                 pca=str(pca_setting), features=fs_name,
                                 n_features=len(cols), primary_axis=primary_axis,
                                 criterion=round(crit, 6), threshold=round(thr, 8),
                                 n_candidates=len(log)))
            print("  seed %d fold %-3s train %2d -> %-16s pca=%-4s feat=%-9s thr=%.4f"
                  % (seed, s, len(tr), model, pca_setting, fs_name, thr), flush=True)


def main():
    app_cfg = yaml.safe_load(io.open(os.path.join(APP, "config.yaml"), encoding="utf-8"))
    d = np.load(NPZ, allow_pickle=True)
    names = [str(x) for x in d["names"]]
    clips = np.array([str(x) for x in d["clips"]])
    groups = np.array([str(x) for x in d["groups"]])
    sessions = np.array([str(x) for x in d["sessions"]])
    primary = np.array([str(x) for x in d["primary"]])
    V = d["vectors"]
    os.makedirs(TABLES, exist_ok=True)

    out_rows, sel_rows, cand_rows = [], [], []
    print(LINE)
    run(app_cfg, V, names, clips, groups, sessions, primary, "clip", out_rows,
        sel_rows, cand_rows)

    W = d["win_vectors"]
    wclip = np.array([str(x) for x in d["win_clip"]])
    idx = {c: i for i, c in enumerate(clips)}
    wgroups = np.array([groups[idx[c]] for c in wclip])
    wsessions = np.array([sessions[idx[c]] for c in wclip])
    wprimary = np.array([primary[idx[c]] for c in wclip])
    print()
    print(LINE)
    run(app_cfg, W, names, wclip, wgroups, wsessions, wprimary, "window",
        out_rows, sel_rows, cand_rows)

    raw = os.path.join(RES, "raw_scores.csv")
    os.makedirs(RES, exist_ok=True)
    with open(raw, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    sel = os.path.join(RES, "selection_log.csv")
    with open(sel, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sel_rows[0].keys()))
        w.writeheader()
        w.writerows(sel_rows)
    cand = os.path.join(RES, "candidates.csv")
    with open(cand, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cand_rows[0].keys()))
        w.writeheader()
        w.writerows(cand_rows)
    print()
    print("wrote", cand, len(cand_rows), "rows")
    print("wrote", raw, len(out_rows), "rows")
    print("wrote", sel, len(sel_rows), "rows")


if __name__ == "__main__":
    main()
