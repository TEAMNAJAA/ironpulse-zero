import os
import sys
import io
import csv
import json
import collections
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
REPO = os.path.dirname(APP)
sys.path.insert(0, REPO)
sys.path.insert(0, APP)
os.environ.setdefault("IRONPULSE_PROJECT_ROOT", APP)

from core import dsp, figures
from appcore import markers, pipeline, calibrate

RES = os.path.join(HERE, "results")
FIGS = os.path.join(RES, "figures")
TABLES = os.path.join(RES, "tables")
RAW = os.path.join(RES, "raw_scores.csv")
SEL = os.path.join(RES, "selection_log.csv")
CAND = os.path.join(RES, "candidates.csv")
NPZ = os.path.join(HERE, "features.npz")
TRACKS = os.path.join(HERE, "tracks.npz")
META = os.path.join(HERE, "tracks_meta.csv")
SETUP_ID = "h4_wall_2026_08_22"
BOOT = 2000
BOOT_SEED = 12345
PRIMARY_LEVEL = "clip"
GROUP_LABEL = {"normal": "ปกติ", "under3g": "ต่ำกว่า 3 กรัม", "3g": "3 กรัม"}
GROUP_EN = {"normal": "normal", "under3g": "under 3 g", "3g": "3 g"}
COLOR = {"normal": "#0072B2", "under3g": "#E69F00", "3g": "#D55E00"}
INSPECTIONS_PER_DAY = 20


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    sv = allv[order]
    r = np.empty(len(allv), float)
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0)
                 / (len(pos) * len(neg)))


def boot_ci(pos, neg, n=BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    out = []
    for _ in range(n):
        out.append(auc(rng.choice(pos, len(pos), replace=True),
                       rng.choice(neg, len(neg), replace=True)))
    out = np.array([v for v in out if np.isfinite(v)])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), out


def roc_points(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    ths = np.unique(np.concatenate([pos, neg]))
    ths = np.concatenate([[-np.inf], ths, [np.inf]])
    tpr = np.array([(pos > t).mean() for t in ths])
    fpr = np.array([(neg > t).mean() for t in ths])
    o = np.argsort(fpr)
    return fpr[o], tpr[o]


def load_rows(path):
    return list(csv.DictReader(io.open(path, encoding="utf-8")))


def clip_scores(rows, level):
    sub = [r for r in rows if r["level"] == level]
    per = collections.defaultdict(list)
    info = {}
    for r in sub:
        key = (r["seed"], r["fold"], r["clip"])
        per[key].append(float(r["norm_score"]))
        info[r["clip"]] = (r["group"], r["session"])
    agg = collections.defaultdict(list)
    for (seed, fold, clip), v in per.items():
        agg[clip].append(float(np.median(v)))
    return {c: float(np.median(v)) for c, v in agg.items()}, info, sub


def fold_metrics(sub, level):
    out = []
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in sub:
        per[(r["seed"], r["fold"])][r["clip"]].append(r)
    for (seed, fold), clips in sorted(per.items()):
        sc, grp, flag = {}, {}, {}
        for c, rs in clips.items():
            sc[c] = float(np.median([float(x["norm_score"]) for x in rs]))
            grp[c] = rs[0]["group"]
            flag[c] = int(sc[c] > 1.0)
        nrm = [c for c in sc if grp[c] == "normal"]
        flt = [c for c in sc if grp[c] != "normal"]
        row = dict(level=level, seed=seed, fold=fold,
                   n_normal=len(nrm), n_fault=len(flt),
                   auc=auc([sc[c] for c in flt], [sc[c] for c in nrm]),
                   far=float(np.mean([flag[c] for c in nrm])) if nrm else float("nan"),
                   tpr_all=float(np.mean([flag[c] for c in flt])) if flt else float("nan"))
        for g in ("under3g", "3g"):
            gg = [c for c in flt if grp[c] == g]
            row["tpr_" + g] = float(np.mean([flag[c] for c in gg])) if gg else float("nan")
        out.append(row)
    return out


def amplitude_table(cfg, cal):
    meta = {r["clip"]: r for r in load_rows(META)}
    d = np.load(TRACKS)
    fs = float(cfg["capture"]["default_fps"])
    rows = {}
    for c, m in meta.items():
        t = d[c].astype(np.float64)
        sigs = markers.differential(t)
        f0 = float(m["f0_hz"])
        ax = m["primary_axis"]
        pk = pipeline.peak_at_order(sigs[ax], fs, cfg, f0)
        a1 = float(pk["amp"]) if pk else float("nan")
        rows[c] = dict(group=m["group"], session=m["session"], f0=f0, axis=ax,
                       a1_px=a1, a1_um=a1 * float(cal["um_per_px"]))
    return rows, d, meta


def fig_a(pooled, info, thr_label, path):
    y = np.array([1 if info[c][0] != "normal" else 0 for c in pooled])
    p = np.array([1 if pooled[c] > 1.0 else 0 for c in pooled])
    cm = np.array([[int(((y == 0) & (p == 0)).sum()), int(((y == 0) & (p == 1)).sum())],
                   [int(((y == 1) & (p == 0)).sum()), int(((y == 1) & (p == 1)).sum())]])
    figures.use()
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    rowsum = cm.sum(axis=1, keepdims=True).astype(float)
    ax.imshow(cm / rowsum, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, "%d\n%.1f%%" % (cm[i, j], 100 * cm[i, j] / rowsum[i, 0]),
                    ha="center", va="center", fontsize=22,
                    color="white" if cm[i, j] / rowsum[i, 0] > 0.55 else "#111111")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["predicted normal", "predicted anomalous"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["true normal", "true unbalanced"])
    ax.set_title("A. Confusion matrix, clip level")
    ax.grid(False)
    fig.text(0.5, -0.02,
             "N = %d clips (%d normal, %d unbalanced) · %s\n"
             "percentages are row-wise · pooled over %d folds x %d seeds"
             % (len(pooled), int((y == 0).sum()), int((y == 1).sum()), thr_label, 4, 5),
             ha="center", fontsize=13)
    figures.save(fig, path)
    return cm


def fig_b(pooled, info, path):
    figures.use()
    fig, ax = plt.subplots(figsize=(12.5, 7))
    bins = np.linspace(min(pooled.values()) * 0.98, max(pooled.values()) * 1.02, 34)
    for g in ("normal", "under3g", "3g"):
        v = [pooled[c] for c in pooled if info[c][0] == g]
        ax.hist(v, bins=bins, alpha=0.72, color=COLOR[g], edgecolor="white",
                label="%s (n=%d)" % (GROUP_EN[g], len(v)))
    ax.axvline(1.0, color="#111111", linestyle="--", linewidth=2.4)
    ax.text(1.0, ax.get_ylim()[1] * 0.94, " threshold", fontsize=14, color="#111111")
    ax.set_xlabel("anomaly score / threshold of its fold")
    ax.set_ylabel("number of clips")
    ax.set_title("B. Score distribution, clip level"
                 + chr(10) + "threshold is set from normal training clips only")
    ax.legend(loc="upper left", fontsize=13)
    figures.save(fig, path)


def fig_c(pooled, info, lo, hi, a, path):
    pos = [pooled[c] for c in pooled if info[c][0] != "normal"]
    neg = [pooled[c] for c in pooled if info[c][0] == "normal"]
    fpr, tpr = roc_points(pos, neg)
    figures.use()
    fig, ax = plt.subplots(figsize=(8.2, 7.4))
    ax.plot([0, 1], [0, 1], color="#999999", linestyle=":", linewidth=2)
    ax.step(fpr, tpr, where="post", color="#0072B2", linewidth=3.0,
            label="AUC = %.3f\n95%% CI %.3f - %.3f" % (a, lo, hi))
    ax.set_xlabel("false alarm rate")
    ax.set_ylabel("detection rate")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("C. ROC, clip level"
                 + chr(10) + "CI from %d bootstrap resamples of clips" % BOOT)
    ax.legend(loc="lower right", fontsize=14)
    figures.save(fig, path)


def fig_d(folds, path):
    figures.use()
    labels, means, sds = [], [], []
    for g in ("3g", "under3g"):
        v = np.array([f["tpr_" + g] for f in folds], float)
        v = v[np.isfinite(v)]
        labels.append(GROUP_EN[g])
        means.append(v.mean())
        sds.append(v.std(ddof=1) if len(v) > 1 else 0.0)
    o = np.argsort(means)[::-1]
    labels = [labels[i] for i in o]
    means = [means[i] for i in o]
    sds = [sds[i] for i in o]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    y = np.arange(len(labels))
    ax.barh(y, means, xerr=sds, color=["#D55E00", "#E69F00"], height=0.55,
            error_kw=dict(ecolor="#333333", capsize=6, lw=2))
    for i, m in enumerate(means):
        ax.text(min(m + sds[i] + 0.03, 1.02), i, "%.2f" % m, va="center", fontsize=16)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("detection rate")
    ax.set_title("D. Detection rate by severity"
                 + chr(10) + "bars are the mean over 20 fold-runs, error bars are 1 sd")
    figures.save(fig, path)


def fig_e(cfg, tracks, meta, path):
    figures.use()
    fs = float(cfg["capture"]["default_fps"])
    grid = np.linspace(0.15, 6.0, 1400)
    curves = collections.defaultdict(list)
    for c in sorted(meta):
        g = meta[c]["group"]
        sigs = markers.differential(tracks[c].astype(np.float64))
        f, A = dsp.amplitude_spectrum(sigs[meta[c]["primary_axis"]], fs,
                                      mode=int(cfg["analysis"]["detrend_order"]),
                                      win=cfg["analysis"]["window_fn"])
        order = f / float(meta[c]["f0_hz"])
        m = (order >= 0.1) & (order <= 6.2)
        curves[g].append(np.interp(grid, order[m], A[m]))
    fig, ax = plt.subplots(figsize=(12.5, 7))
    for g in ("normal", "under3g", "3g"):
        C = np.vstack(curves[g])
        lo = np.percentile(C, 10, axis=0)
        hi = np.percentile(C, 90, axis=0)
        md = np.percentile(C, 50, axis=0)
        ax.fill_between(grid, np.maximum(lo, 1e-7), np.maximum(hi, 1e-7),
                        color=COLOR[g], alpha=0.22, linewidth=0)
        ax.semilogy(grid, np.maximum(md, 1e-7), color=COLOR[g], linewidth=2.6,
                    label="%s (n=%d)" % (GROUP_EN[g], C.shape[0]))
    for k in (1, 2, 3, 4, 5):
        ax.axvline(k, color="#cccccc", linewidth=1.0, zorder=0)
    ax.set_xlabel("Order (multiples of shaft speed)")
    ax.set_ylabel("amplitude (px)")
    ax.set_xlim(0.15, 6.0)
    ax.set_title("E. Order spectrum of the axis carrying the signal"
                 + chr(10)
                 + "thick line is the group median, band is the 10th to 90th percentile")
    ax.legend(loc="upper right", fontsize=13)
    figures.save(fig, path)


def fig_f(cands, level, path):
    rows = [r for r in cands if r["level"] == level]
    best = collections.defaultdict(list)
    for r in rows:
        best[(r["seed"], r["fold"], r["model"])].append(float(r["criterion"]))
    per_model = collections.defaultdict(list)
    for (seed, fold, model), v in best.items():
        per_model[model].append(min(v))
    figures.use()
    names = sorted(per_model, key=lambda m: np.mean(per_model[m]))
    fig, ax = plt.subplots(figsize=(11, 6))
    data = [per_model[m] for m in names]
    bp = ax.boxplot(data, vert=True, patch_artist=True, widths=0.55)
    for patch, c in zip(bp["boxes"], figures.OKABE_ITO):
        patch.set_facecolor(c)
        patch.set_alpha(0.65)
    for med in bp["medians"]:
        med.set_color("#111111")
        med.set_linewidth(2.4)
    ax.set_xticklabels(names, fontsize=13)
    ax.set_ylabel("inner-loop criterion (lower is tighter)")
    ax.set_title("F. Inner-loop comparison of the four models"
                 + chr(10) + "best configuration of each model in each of 20 fold-runs")
    figures.save(fig, path)
    return {m: (float(np.mean(per_model[m])), float(np.std(per_model[m], ddof=1)))
            for m in names}


def main():
    cfg = yaml.safe_load(io.open(os.path.join(APP, "config.yaml"), encoding="utf-8"))
    cals = calibrate.load(os.path.join(APP, cfg["scale"]["calibration_file"]))
    cal = calibrate.require(cals, SETUP_ID)
    os.makedirs(FIGS, exist_ok=True)
    os.makedirs(TABLES, exist_ok=True)
    raw = load_rows(RAW)
    sel = load_rows(SEL)
    cands = load_rows(CAND)

    summary = {}
    for level in ("clip", "window"):
        pooled, info, sub = clip_scores(raw, level)
        folds = fold_metrics(sub, level)
        pos = [pooled[c] for c in pooled if info[c][0] != "normal"]
        neg = [pooled[c] for c in pooled if info[c][0] == "normal"]
        a = auc(pos, neg)
        lo, hi, _ = boot_ci(pos, neg)
        summary[level] = dict(pooled=pooled, info=info, folds=folds,
                              auc=a, lo=lo, hi=hi)
        print("%-7s pooled AUC %.4f (95%% CI %.4f - %.4f)  fold AUC %.4f +- %.4f"
              % (level, a, lo, hi,
                 np.mean([f["auc"] for f in folds]),
                 np.std([f["auc"] for f in folds], ddof=1)))

    lv = summary[PRIMARY_LEVEL]
    pooled, info, folds = lv["pooled"], lv["info"], lv["folds"]
    thr_lab = "decision at score / threshold > 1.0"
    cm = fig_a(pooled, info, thr_lab, os.path.join(FIGS, "A_confusion_matrix.png"))
    fig_b(pooled, info, os.path.join(FIGS, "B_score_distribution.png"))
    fig_c(pooled, info, lv["lo"], lv["hi"], lv["auc"],
          os.path.join(FIGS, "C_roc_curve.png"))
    fig_d(folds, os.path.join(FIGS, "D_detection_by_severity.png"))
    amps, tracks, meta = amplitude_table(cfg, cal)
    fig_e(cfg, tracks, meta, os.path.join(FIGS, "E_order_spectra.png"))
    model_stats = fig_f(cands, PRIMARY_LEVEL, os.path.join(FIGS, "F_model_comparison.png"))

    with open(os.path.join(TABLES, "table1_by_severity.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "severity", "n_clips", "detection_rate_mean",
                    "detection_rate_sd", "score_mean", "score_sd",
                    "a1_px_mean", "a1_um_mean", "f0_hz_mean"])
        for g in ("normal", "under3g", "3g"):
            cl = [c for c in pooled if info[c][0] == g]
            sc = np.array([pooled[c] for c in cl])
            key = "far" if g == "normal" else "tpr_" + g
            v = np.array([fo[key] for fo in folds], float)
            v = v[np.isfinite(v)]
            w.writerow([GROUP_EN[g], g, len(cl), round(v.mean(), 4),
                        round(v.std(ddof=1), 4), round(sc.mean(), 4),
                        round(sc.std(ddof=1), 4),
                        round(np.mean([amps[c]["a1_px"] for c in cl]), 5),
                        round(np.mean([amps[c]["a1_um"] for c in cl]), 2),
                        round(np.mean([amps[c]["f0"] for c in cl]), 4)])

    with open(os.path.join(TABLES, "table2_summary.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value", "ci_low", "ci_high", "note"])
        w.writerow(["roc_auc_pooled", round(lv["auc"], 4), round(lv["lo"], 4),
                    round(lv["hi"], 4), "%d bootstrap resamples at clip level" % BOOT])
        af = np.array([fo["auc"] for fo in folds])
        w.writerow(["roc_auc_fold_mean", round(af.mean(), 4), "", "",
                    "sd %.4f over %d fold-runs" % (af.std(ddof=1), len(af))])
        tp = np.array([fo["tpr_all"] for fo in folds])
        w.writerow(["detection_rate_all", round(tp.mean(), 4), "", "",
                    "sd %.4f" % tp.std(ddof=1)])
        for g in ("3g", "under3g"):
            v = np.array([fo["tpr_" + g] for fo in folds], float)
            w.writerow(["detection_rate_" + g, round(np.nanmean(v), 4), "", "",
                        "sd %.4f" % np.nanstd(v, ddof=1)])
        fa = np.array([fo["far"] for fo in folds])
        w.writerow(["false_alarm_rate", round(fa.mean(), 4), "", "",
                    "normal clips only, sd %.4f" % fa.std(ddof=1)])
        w.writerow(["px_per_mm", round(cal["px_per_mm"], 4), "", "",
                    "ruler, %.2f um/px" % cal["um_per_px"]])
        w.writerow(["detection_floor_um",
                    round(calibrate.detection_floor_um(
                        cal, float(cfg["scale"]["flow_resolution_px"])), 2), "", "",
                    "flow resolution %.3f px" % float(cfg["scale"]["flow_resolution_px"])])

    with open(os.path.join(TABLES, "table3_fold_metrics.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(folds[0].keys()))
        w.writeheader()
        for fo in summary["clip"]["folds"] + summary["window"]["folds"]:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v)
                        for k, v in fo.items()})

    with open(os.path.join(TABLES, "table4_model_selection.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "criterion_mean", "criterion_sd", "times_selected_clip",
                    "times_selected_window"])
        cnt_c = collections.Counter(r["model"] for r in sel if r["level"] == "clip")
        cnt_w = collections.Counter(r["model"] for r in sel if r["level"] == "window")
        for m, (mu, sd) in sorted(model_stats.items(), key=lambda kv: kv[1][0]):
            w.writerow([m, round(mu, 5), round(sd, 5), cnt_c.get(m, 0), cnt_w.get(m, 0)])

    with open(os.path.join(TABLES, "table5_per_clip.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clip", "group", "session", "f0_hz", "primary_axis",
                    "a1_px", "a1_um", "pooled_score", "flagged"])
        for c in sorted(pooled):
            a = amps[c]
            w.writerow([c, a["group"], a["session"], round(a["f0"], 4), a["axis"],
                        round(a["a1_px"], 5), round(a["a1_um"], 2),
                        round(pooled[c], 5), int(pooled[c] > 1.0)])

    js = os.path.join(RES, "summary.json")
    io.open(js, "w", encoding="utf-8").write(json.dumps(dict(
        auc=lv["auc"], ci=[lv["lo"], lv["hi"]], cm=cm.tolist(),
        far=float(np.mean([f["far"] for f in folds])),
        tpr_3g=float(np.nanmean([f["tpr_3g"] for f in folds])),
        tpr_under3g=float(np.nanmean([f["tpr_under3g"] for f in folds])),
        window_auc=summary["window"]["auc"],
        um_per_px=cal["um_per_px"]), ensure_ascii=False, indent=2))
    print()
    print("confusion matrix:", cm.tolist())
    print("wrote figures to", FIGS)
    print("wrote tables to", TABLES)


if __name__ == "__main__":
    main()
