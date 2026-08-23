import os, sys, csv, json
import numpy as np
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)
sys.path.insert(0, os.path.dirname(_PROJ))
from experiments.common import CFG

T = CFG["paths"]["tables"]
OUT = os.path.join(CFG["paths"]["results"], "summary.json")


def load(name):
    p = os.path.join(T, name)
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return []
    return list(csv.DictReader(open(p, encoding="utf-8")))


def f(r, k, d=np.nan):
    try:
        v = r.get(k, "")
        return float(v) if v not in ("", None) else d
    except (TypeError, ValueError):
        return d


def one(rows, **kw):
    for r in rows:
        if all(str(r.get(k, "")) == str(v) for k, v in kw.items()):
            return r
    return None


def match(rows, **kw):
    return [r for r in rows if all(str(r.get(k, "")) == str(v) for k, v in kw.items())]


S = {}

e1 = load("e1_baseline_overall.csv")
S["E1"] = {}
for ds in sorted(set(r["dataset"] for r in e1)):
    d = {}
    for dom in ["acceleration", "displacement"]:
        r = one(e1, dataset=ds, domain=dom)
        if r:
            d[dom] = dict(auc=f(r, "roc_auc_mean"), sd=f(r, "roc_auc_std"),
                          pr=f(r, "pr_auc_mean"), f1=f(r, "f1_mean"), far=f(r, "far_mean"),
                          n_windows=int(f(r, "n_windows", 0)), n_files=int(f(r, "n_files", 0)),
                          folds=int(f(r, "n_folds", 0)))
    if "acceleration" in d and "displacement" in d:
        a, b = d["acceleration"]["auc"], d["displacement"]["auc"]
        d["domain_cost_abs"] = round(a - b, 4)
        d["domain_cost_pct"] = round(100.0 * (a - b) / a, 1) if a else None
    S["E1"][ds] = d

e1p = load("e1_baseline_per_label.csv")
S["E1_per_label"] = {}
for ds in sorted(set(r["dataset"] for r in e1p)):
    S["E1_per_label"][ds] = {}
    for dom in ["acceleration", "displacement"]:
        S["E1_per_label"][ds][dom] = {
            r["label"]: round(f(r, "roc_auc_mean"), 4)
            for r in match(e1p, dataset=ds, domain=dom)}

S["E2"] = {}
for suffix in ["", "_cwru", "_mafaulda"]:
    rows = load("e2_sampling_rate_overall%s.csv" % suffix)
    for ds in sorted(set(r["dataset"] for r in rows)):
        sub = match(rows, dataset=ds)
        full = [r for r in sub if f(r, "fs_hz") > 40000]
        ref = f(full[0], "roc_auc_mean") if full else np.nan
        d = dict(full_reference=round(ref, 4) if np.isfinite(ref) else None, by_mode={})
        for mode in CFG["camera"]["decimation_modes"]:
            pts = sorted([r for r in sub if r["mode"] == mode and f(r, "fs_hz") <= 40000],
                         key=lambda r: -f(r, "fs_hz"))
            curve = {int(f(r, "fs_hz")): dict(auc=round(f(r, "roc_auc_mean"), 4),
                                              sd=round(f(r, "roc_auc_std"), 4))
                     for r in pts}
            r240 = one(sub, mode=mode, fs_hz="240.0")
            entry = dict(curve=curve)
            if r240 is not None and np.isfinite(ref) and ref:
                v = f(r240, "roc_auc_mean")
                entry["at_240"] = round(v, 4)
                entry["at_240_sd"] = round(f(r240, "roc_auc_std"), 4)
                entry["drop_from_full_abs"] = round(ref - v, 4)
                entry["drop_from_full_pct"] = round(100.0 * (ref - v) / ref, 1)
            d["by_mode"][mode] = entry
        S["E2"][ds] = d

e2p = load("e2_sampling_rate_per_label.csv") or load("e2_sampling_rate_per_label_mafaulda.csv")
S["E2_per_label_240_boxcar"] = {}
S["E2_per_label_full"] = {}
for ds in sorted(set(r["dataset"] for r in e2p)):
    S["E2_per_label_240_boxcar"][ds] = {
        r["label"]: round(f(r, "roc_auc_mean"), 4)
        for r in match(e2p, dataset=ds, mode="boxcar", fs_hz="240.0")}
    S["E2_per_label_full"][ds] = {
        r["label"]: round(f(r, "roc_auc_mean"), 4)
        for r in e2p if r["dataset"] == ds and f(r, "fs_hz") > 40000}

e2b = load("e2b_speed_band.csv")
S["E2b_speed_band"] = {}
for fs in sorted(set(f(r, "fs_hz") for r in e2b)):
    key = "%g" % fs
    S["E2b_speed_band"][key] = [
        dict(band="%s-%s Hz" % (r["band_lo"], r["band_hi"]),
             median_f0=f(r, "median_f0_hz"), max_order=f(r, "max_usable_order"),
             auc=round(f(r, "roc_auc"), 4), n=int(f(r, "n_windows", 0)))
        for r in e2b if f(r, "fs_hz") == fs]

e3 = load("e3_detectors_overall.csv")
S["E3"] = {}
for ds in sorted(set(r["dataset"] for r in e3)):
    S["E3"][ds] = {r["detector"]: dict(auc=round(f(r, "roc_auc_mean"), 4),
                                       sd=round(f(r, "roc_auc_std"), 4),
                                       f1=round(f(r, "f1_mean"), 4),
                                       far=round(f(r, "far_mean"), 4),
                                       seconds=f(r, "fit_score_s"))
                   for r in match(e3, dataset=ds)}

e4 = load("e4_features.csv")
S["E4"] = {}
for ds in sorted(set(r["dataset"] for r in e4)):
    S["E4"][ds] = {r["variant"]: dict(auc=round(f(r, "roc_auc_mean"), 4),
                                      n_features=int(f(r, "n_features", 0)),
                                      delta=f(r, "delta_auc"))
                   for r in match(e4, dataset=ds)}

e5 = load("e5_transfer.csv")
S["E5"] = {}
for r in e5:
    S["E5"].setdefault(r["dataset"], {}).setdefault(r["split"], {})[r["axis"]] = dict(
        auc=round(f(r, "roc_auc"), 4), far=round(f(r, "far"), 4),
        n_train=int(f(r, "n_train", 0)))

e6 = load("e6_noise_overall.csv")
S["E6"] = {}
for ds in sorted(set(r["dataset"] for r in e6)):
    d = {}
    cl = one(e6, dataset=ds, noise_type="none")
    if cl:
        d["clean"] = round(f(cl, "roc_auc_mean"), 4)
    for nt in ["white", "shake"]:
        d[nt] = {int(f(r, "snr_db")): round(f(r, "roc_auc_mean"), 4)
                 for r in match(e6, dataset=ds, noise_type=nt) if np.isfinite(f(r, "snr_db"))}
    S["E6"][ds] = d

e7 = load("e7_resolution_overall.csv")
S["E7"] = {}
for ds in sorted(set(r["dataset"] for r in e7)):
    sub = match(e7, dataset=ds)
    base = [r for r in sub if f(r, "quant_px") == 0]
    d = dict(no_quantisation=round(f(base[0], "roc_auc_mean"), 4) if base else None,
             median_rms_px=f(base[0], "median_rms_px") if base else None, steps={})
    for r in sorted([x for x in sub if f(x, "quant_px") > 0], key=lambda x: f(x, "quant_px")):
        d["steps"]["%g" % f(r, "quant_px")] = dict(
            auc=round(f(r, "roc_auc_mean"), 4), sd=round(f(r, "roc_auc_std"), 4),
            um=f(r, "quant_um"), dead_frac=f(r, "dead_window_frac"))
    if base and d["steps"]:
        ref = d["no_quantisation"]
        for k, v in d["steps"].items():
            v["retained_pct"] = round(100.0 * v["auc"] / ref, 1) if ref else None
    S["E7"][ds] = d

e8 = load("e8_trend_summary.csv")
S["E8"] = [dict(test=r["test"], condition=r["condition"], featureset=r["featureset"],
                domain=r["domain"], fs_hz=f(r, "fs_hz"),
                alarm_h=f(r, "alarm_hours"), failure_h=f(r, "failure_hours"),
                lead_h=f(r, "lead_time_hours"),
                lead_pct=f(r, "lead_time_pct_of_life"),
                n=int(f(r, "n_snapshots", 0))) for r in e8]

opt = CFG["optics"]
S["scale"] = dict(pixels_per_meter=opt["pixels_per_meter"],
                  px_for_50um=round(50e-6 * opt["pixels_per_meter"], 5),
                  px_for_10um=round(10e-6 * opt["pixels_per_meter"], 5),
                  um_per_px=round(1e6 / opt["pixels_per_meter"], 2))

json.dump(S, open(OUT, "w", encoding="utf-8"), indent=2, default=str)
print(json.dumps(S, indent=2, default=str))
print("\nwrote", OUT)
