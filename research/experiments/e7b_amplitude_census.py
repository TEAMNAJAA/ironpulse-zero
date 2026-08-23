import os, sys
import numpy as np
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)
sys.path.insert(0, os.path.dirname(_PROJ))
from experiments.common import CFG, maf_index_all, cwru_index_48k, write_csv, table
from core import pipeline, features as F

FS = float(CFG["camera"]["primary_fs_hz"])
EXP = CFG["camera"]["default_exposure_sec"]
PPM = CFG["optics"]["pixels_per_meter"]
BANDS = [(10, 20), (20, 30), (30, 40), (40, 50), (50, 65)]
QS = CFG["camera"]["quantization_px"]


def pct(a, p):
    return float(np.percentile(a, p)) if len(a) else float("nan")


def run(name, index):
    X, Xr, meta = pipeline.build(index, CFG, fs_out=FS, mode="boxcar", exposure_sec=EXP,
                                 domain="displacement", progress=500)
    if not len(meta):
        return []
    labs = np.array([m["label"] for m in meta])
    f0 = np.array([m["f0_hz"] for m in meta])
    rms = Xr[:, F.RAW.index("rms_px")]
    a1 = Xr[:, F.RAW.index("A1_px")]
    rows = []
    for lab in sorted(set(labs.tolist())):
        for lo, hi in [(0, 999)] + BANDS:
            sel = (labs == lab) & (f0 >= lo) & (f0 < hi)
            if sel.sum() < 20:
                continue
            r = dict(dataset=name, label=lab,
                     speed_band=("all" if lo == 0 else "%d-%d Hz" % (lo, hi)),
                     n_windows=int(sel.sum()),
                     median_f0_hz=round(float(np.median(f0[sel])), 2),
                     rms_px_p10=round(pct(rms[sel], 10), 5),
                     rms_px_median=round(pct(rms[sel], 50), 5),
                     rms_px_p90=round(pct(rms[sel], 90), 5),
                     rms_um_median=round(pct(rms[sel], 50) / PPM * 1e6, 2),
                     a1_px_median=round(pct(a1[sel], 50), 5),
                     a1_um_median=round(pct(a1[sel], 50) / PPM * 1e6, 2))
            for q in QS:
                r["frac_rms_above_%g_px" % q] = round(float(np.mean(rms[sel] > q)), 3)
            rows.append(r)
    return rows


if __name__ == "__main__":
    O = []
    for nm, idx in [("mafaulda", maf_index_all()), ("cwru_48k", cwru_index_48k())]:
        print("===", nm, len(idx), "files", flush=True)
        rr = run(nm, idx)
        O += rr
        for r in rr:
            if r["speed_band"] == "all":
                print("  %-26s med_rms=%8.5f px (%7.2f um)  1x=%8.5f px  >0.01px:%.2f  >0.05px:%.2f"
                      % (r["label"], r["rms_px_median"], r["rms_um_median"],
                         r["a1_px_median"], r["frac_rms_above_0.01_px"],
                         r["frac_rms_above_0.05_px"]), flush=True)
    write_csv(table("t", "e7b_amplitude_census.csv"), O)
