import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def make_folds(file_ids, labels, n_folds, seed):
    files = {}
    for fid, lab in zip(file_ids, labels):
        files[fid] = lab
    normals = sorted([f for f, l in files.items() if l == "normal"])
    others = sorted([f for f, l in files.items() if l != "normal"])
    k = int(min(n_folds, len(normals))) if normals else int(n_folds)
    k = max(k, 2)
    rng = np.random.default_rng(seed)

    def chunk(items):
        items = list(items)
        rng.shuffle(items)
        out = [[] for _ in range(k)]
        for i, it in enumerate(items):
            out[i % k].append(it)
        return out

    nf = chunk(normals)
    by_lab = {}
    for f in others:
        by_lab.setdefault(files[f], []).append(f)
    of = [[] for _ in range(k)]
    for lab in sorted(by_lab):
        for i, part in enumerate(chunk(by_lab[lab])):
            of[i].extend(part)
    return k, [sorted(nf[i] + of[i]) for i in range(k)]


def binary_metrics(y, s, thr):
    out = {}
    if len(set(y.tolist())) < 2:
        return dict(roc_auc=np.nan, pr_auc=np.nan, f1=np.nan,
                    far=float(np.mean(s[y == 0] > thr)) if (y == 0).any() else np.nan,
                    n_norm=int((y == 0).sum()), n_anom=int((y == 1).sum()))
    out["roc_auc"] = float(roc_auc_score(y, s))
    out["pr_auc"] = float(average_precision_score(y, s))
    out["f1"] = float(f1_score(y, (s > thr).astype(int), zero_division=0))
    out["far"] = float(np.mean(s[y == 0] > thr))
    out["n_norm"] = int((y == 0).sum())
    out["n_anom"] = int((y == 1).sum())
    return out


def run_cv(X, meta, detector_cls, cfg, n_folds=None, per_label=True, seed=None):
    seed = cfg["seed"] if seed is None else seed
    fids = np.array([m["file_id"] for m in meta])
    labs = np.array([m["label"] for m in meta])
    nf = n_folds or cfg["evaluate"]["n_folds"]
    k, folds = make_folds(fids, labs, nf, seed)
    rows = []
    per = []
    for i in range(k):
        test_files = set(folds[i])
        te = np.array([f in test_files for f in fids])
        tr = (~te) & (labs == "normal")
        if tr.sum() < 5 or te.sum() == 0:
            continue
        det = detector_cls(cfg, seed=seed).fit(X[tr])
        thr = det.threshold()
        s = det.score(X[te])
        y = (labs[te] != "normal").astype(int)
        m = binary_metrics(y, s, thr)
        m.update(fold=i, n_train=int(tr.sum()))
        rows.append(m)
        if per_label:
            lt = labs[te]
            base = s[y == 0]
            for lab in sorted(set(lt[y == 1])):
                sel = lt == lab
                if not sel.any() or not len(base):
                    continue
                ss = np.r_[base, s[sel]]
                yy = np.r_[np.zeros(len(base)), np.ones(int(sel.sum()))]
                mm = binary_metrics(yy, ss, thr)
                mm.update(fold=i, label=lab)
                per.append(mm)
    return rows, per


def agg(rows, keys=("roc_auc", "pr_auc", "f1", "far")):
    out = {}
    for kk in keys:
        v = np.array([r[kk] for r in rows if kk in r and not np.isnan(r[kk])], dtype=float)
        out[kk + "_mean"] = float(v.mean()) if len(v) else np.nan
        out[kk + "_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    out["n_folds"] = len(rows)
    return out
