import numpy as np
from . import dsp

DIMENSIONLESS = ["r1", "r2", "r3", "r05", "rbp", "E_sub", "E_1x", "E_mid", "E_hi",
                 "harm", "centroid", "crest", "kurt"]
RAW = ["rms_px", "peak_px", "A05_px", "A1_px", "A2_px", "A3_px", "f0_hz", "nyq_order"]

GROUPS = {
    "orders": ["r1", "r2", "r3", "r05"],
    "bands": ["E_sub", "E_1x", "E_mid", "E_hi"],
    "shape": ["harm", "centroid"],
    "time": ["crest", "kurt"],
    "blade": ["rbp"],
}


def extract(x, fs, f0_hz, cfg):
    fcfg = cfg["features"]
    order_floor = cfg["integration"].get("highpass_order_floor", 0.1)
    hw = fcfg["peak_search_halfwidth_order"]
    f, A = dsp.amplitude_spectrum(x, fs, mode=cfg["signal"]["detrend"],
                                  win=cfg["signal"]["window_fn"])
    out = {k: 0.0 for k in DIMENSIONLESS}
    raw = {k: 0.0 for k in RAW}
    ts = dsp.time_stats(x)
    raw["rms_px"] = ts["rms"]
    raw["peak_px"] = ts["peak"]
    raw["f0_hz"] = float(f0_hz) if f0_hz else 0.0
    out["crest"] = ts["crest"]
    out["kurt"] = ts["kurt"]
    if not f0_hz or f0_hz <= 0:
        return out, raw, {k: False for k in DIMENSIONLESS}
    orders = f / float(f0_hz)
    nyq_order = (fs / 2.0) / float(f0_hz)
    raw["nyq_order"] = nyq_order
    band = orders >= order_floor
    if not band.any():
        return out, raw, {k: False for k in DIMENSIONLESS}
    Ab = A[band]
    ob = orders[band]
    e_tot = float(np.sum(Ab ** 2))
    rms_band = float(np.sqrt(e_tot / 2.0))
    avail = {k: True for k in DIMENSIONLESS}
    amps = {}
    for k, name in [(0.5, "r05"), (1.0, "r1"), (2.0, "r2"), (3.0, "r3")]:
        if k + hw <= nyq_order:
            a, ok = dsp.peak_amplitude_at_order(ob, Ab, k, hw)
        else:
            a, ok = 0.0, False
        amps[name] = a
        avail[name] = ok
        out[name] = (a / rms_band) if rms_band > 0 and ok else 0.0
    raw["A05_px"], raw["A1_px"] = amps["r05"], amps["r1"]
    raw["A2_px"], raw["A3_px"] = amps["r2"], amps["r3"]
    nb = fcfg.get("blade_count")
    if nb:
        if nb + hw <= nyq_order:
            a, ok = dsp.peak_amplitude_at_order(ob, Ab, float(nb), hw)
        else:
            a, ok = 0.0, False
        avail["rbp"] = ok
        out["rbp"] = (a / rms_band) if rms_band > 0 and ok else 0.0
    else:
        avail["rbp"] = False
        out["rbp"] = 0.0
    for name, (lo, hi) in fcfg["bands"].items():
        if lo >= nyq_order:
            out[name] = 0.0
            avail[name] = False
            continue
        hi_eff = min(hi, nyq_order)
        e, ok = dsp.band_energy(ob, Ab, lo, hi_eff)
        out[name] = (e / e_tot) if e_tot > 0 and ok else 0.0
        avail[name] = ok
    a1 = amps["r1"]
    out["harm"] = ((amps["r2"] + amps["r3"]) / a1) if a1 > 0 else 0.0
    avail["harm"] = avail["r1"] and (avail["r2"] or avail["r3"])
    out["centroid"] = dsp.spectral_centroid_order(ob, Ab, lo=order_floor,
                                                  hi=min(fcfg["max_order"], nyq_order))
    return out, raw, avail


def vector(feat, names=None):
    names = names or DIMENSIONLESS
    return np.array([feat[n] for n in names], dtype=np.float64)
