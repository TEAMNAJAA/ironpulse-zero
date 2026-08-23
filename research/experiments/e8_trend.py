import os, sys
import numpy as np
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)
sys.path.insert(0, os.path.dirname(_PROJ))
from experiments.common import CFG, write_csv, table
from core import ims, camera_sim as cs, features as F, detectors, signals as S

TRAIN_FRAC = 0.20
PERSIST = 3
SHUTDOWN_REL_RMS = 0.05
FEATURESETS = ["dimensionless", "dimensionless_plus_amplitude"]


def build_series(test, fs_out, mode, domain, exposure_sec=None, stride=1):
    idx = ims.index(test)
    if not idx:
        return None
    idx = idx[::stride]
    Xd, Xa, hs = [], [], []
    for r in idx:
        try:
            x, fs, f0 = ims.load(r)
        except Exception:
            continue
        a = np.asarray(x, dtype=np.float64) * CFG["integration"]["g0"]
        a, _ = cs.rebase(a, fs, cs.BASE_FS)
        if domain == "displacement":
            sig = cs.accel_to_displacement(
                a, cs.BASE_FS, highpass_hz=CFG["integration"]["highpass_hz"],
                hp_order=CFG["integration"]["highpass_order"], g0=None,
                f0_hz=f0, order_floor=CFG["integration"]["highpass_order_floor"])
        else:
            sig = a
        y = cs.decimate(sig, cs.BASE_FS, fs_out, mode=mode, exposure_sec=exposure_sec)
        if domain == "displacement":
            y = cs.to_pixels(y, CFG["optics"]["pixels_per_meter"])
        for _, w in S.windows(y, fs_out, CFG["signal"]["window_sec"], CFG["signal"]["hop_sec"]):
            f, raw, av = F.extract(w, fs_out, f0, CFG)
            Xd.append(F.vector(f))
            Xa.append(np.array([raw[k] for k in F.RAW], dtype=np.float64))
            hs.append(r["hours_from_start"])
    if not Xd:
        return None
    return np.vstack(Xd), np.vstack(Xa), np.array(hs)


def make_X(Xd, Xa, featureset):
    if featureset == "dimensionless":
        return Xd
    amp = Xa[:, [F.RAW.index("rms_px"), F.RAW.index("peak_px"), F.RAW.index("A1_px")]]
    return np.hstack([Xd, np.log10(np.maximum(amp, 1e-12))])


def analyse_one(Xd, Xa, h, featureset, meta):
    X = make_X(Xd, Xa, featureset)
    t_fail = float(h.max())
    ntr = max(int(TRAIN_FRAC * len(h)), 10)
    tr = np.zeros(len(h), bool)
    tr[:ntr] = True
    det = detectors.MahalanobisDetector(CFG).fit(X[tr])
    thr = det.threshold()
    s = det.score(X)
    over = s > thr
    alarm_h, run = None, 0
    for i in range(ntr, len(h)):
        run = run + 1 if over[i] else 0
        if run >= PERSIST:
            alarm_h = float(h[i - PERSIST + 1])
            break
    lead = (t_fail - alarm_h) if alarm_h is not None else None
    row = dict(meta)
    row.update(featureset=featureset, n_snapshots=int(len(h)),
               train_until_hours=round(float(h[ntr - 1]), 2),
               failure_hours=round(t_fail, 2),
               alarm_hours=round(alarm_h, 2) if alarm_h is not None else "",
               lead_time_hours=round(lead, 2) if lead is not None else "",
               lead_time_pct_of_life=round(100.0 * lead / t_fail, 1) if lead else "",
               threshold=round(thr, 4),
               score_healthy_mean=round(float(s[tr].mean()), 4),
               score_final_mean=round(float(s[h >= 0.9 * t_fail].mean()), 4))
    series = [dict(test=meta["test"], condition=meta["condition"], featureset=featureset,
                   hours=round(float(a), 4), score=round(float(b), 6),
                   threshold=round(thr, 6)) for a, b in zip(h, s)]
    return row, series


if __name__ == "__main__":
    EXP = CFG["camera"]["default_exposure_sec"]
    CONDS = [
        (cs.BASE_FS, "ideal", "acceleration", None, "accel_full"),
        (cs.BASE_FS, "ideal", "displacement", None, "disp_full_48kHz"),
        (240.0, "ideal", "displacement", None, "disp_240Hz_ideal"),
        (240.0, "boxcar", "displacement", EXP, "disp_240Hz_boxcar"),
        (240.0, "naive", "displacement", None, "disp_240Hz_naive"),
    ]
    rows, allseries, ampseries = [], [], []
    for test, stride in [("2nd_test", 1), ("3rd_test", 2)]:
        if not ims.test_dir(test):
            print("skip", test, "(not extracted)", flush=True)
            continue
        for fs_out, mode, domain, exp, tag in CONDS:
            got = build_series(test, fs_out, mode, domain, exp, stride=stride)
            if got is None:
                continue
            Xd, Xa, h = got
            rms = Xa[:, F.RAW.index("rms_px")]
            med = float(np.median(rms))
            live = rms > SHUTDOWN_REL_RMS * med
            Xd, Xa, h = Xd[live], Xa[live], h[live]
            if len(h) < 30:
                continue
            meta = dict(test=test, condition=tag, domain=domain, fs_hz=fs_out, mode=mode)
            for fsname in FEATURESETS:
                r, ser = analyse_one(Xd, Xa, h, fsname, meta)
                rows.append(r)
                allseries.extend(ser)
                print("  %-9s %-18s %-28s alarm=%-7s failure=%.1f h  LEAD=%s h" % (
                    test, tag, fsname, r["alarm_hours"], r["failure_hours"],
                    r["lead_time_hours"]), flush=True)
            amp = Xa[:, F.RAW.index("rms_px")]
            ampseries.extend([dict(test=test, condition=tag, hours=round(float(a), 4),
                                   rms=float("%.8g" % b)) for a, b in zip(h, amp)])
    write_csv(table("t", "e8_trend_summary.csv"), rows)
    write_csv(table("t", "e8_trend_series.csv"), allseries)
    write_csv(table("t", "e8_amplitude_series.csv"), ampseries)
