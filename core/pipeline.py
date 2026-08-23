import os, json
import numpy as np
from . import signals as S, camera_sim as cs, features as F, config

CACHE = os.path.join(config.ROOT, "datasets", "cache", "accel48k")
DROPPED = []


def _cache_paths(rec):
    d = os.path.join(CACHE, rec["source"])
    os.makedirs(d, exist_ok=True)
    b = os.path.join(d, rec["file_id"])
    return b + ".npy", b + ".json"


def accel_for(rec, cfg, force=False):
    npy, js = _cache_paths(rec)
    if os.path.exists(npy) and os.path.exists(js) and not force:
        meta = json.load(open(js, encoding="utf-8"))
        return np.load(npy).astype(np.float64), cs.BASE_FS, meta
    loader = S.cwru_load if rec["source"] == "cwru" else S.maf_load
    try:
        d = loader(rec)
    except Exception as e:
        DROPPED.append((rec["file_id"], "load_error:%s" % e))
        return None, None, None
    if d is None:
        DROPPED.append((rec["file_id"], "loader_returned_none"))
        return None, None, None
    if d["f0_hz"] is None or not np.isfinite(d["f0_hz"]) or d["f0_hz"] <= 0:
        DROPPED.append((rec["file_id"], "no_f0"))
        return None, None, None
    a = np.asarray(d["signal"], dtype=np.float64) * cfg["integration"]["g0"]
    a, _ = cs.rebase(a, d["fs"], cs.BASE_FS)
    meta = dict(file_id=d["file_id"], label=d["label"], severity=d["severity"],
                machine_id=d["machine_id"], source=d["source"], channel=d["channel"],
                f0_hz=d["f0_hz"], f0_method=d["f0_method"], fs_orig=d["fs"],
                fs_base=cs.BASE_FS, n=len(a))
    np.save(npy, a.astype(np.float32))
    json.dump(meta, open(js, "w", encoding="utf-8"))
    return a, cs.BASE_FS, meta


def signal_for(rec, cfg, domain="displacement"):
    a, fsb, m = accel_for(rec, cfg)
    if a is None:
        return None, None, None
    if domain == "acceleration":
        return a, fsb, m
    d = cs.accel_to_displacement(
        a, fsb,
        highpass_hz=cfg["integration"]["highpass_hz"],
        hp_order=cfg["integration"]["highpass_order"],
        g0=None, f0_hz=m["f0_hz"],
        order_floor=cfg["integration"]["highpass_order_floor"])
    return d, fsb, m


def build(index, cfg, fs_out=None, mode="ideal", exposure_sec=None, quant_px=None,
          snr_db=None, noise_type=None, seed=None, progress=None,
          domain="displacement", scale_to_pixels=True, f0_override=None):
    seed = cfg["seed"] if seed is None else seed
    fs_out = fs_out or cs.BASE_FS
    X, Xr, meta = [], [], []
    for i, rec in enumerate(index):
        sig, fsb, m = signal_for(rec, cfg, domain=domain)
        if sig is None:
            continue
        rng = np.random.default_rng(abs(hash((seed, m["file_id"]))) % (2 ** 32))
        y = cs.decimate(sig, fsb, fs_out, mode=mode, exposure_sec=exposure_sec)
        if domain == "displacement" and scale_to_pixels:
            y = cs.to_pixels(y, cfg["optics"]["pixels_per_meter"])
        if snr_db is not None and noise_type:
            if noise_type == "white":
                y = cs.add_white_noise(y, snr_db, rng)
            elif noise_type == "shake":
                y = cs.add_shake(y, fs_out, snr_db, cfg["camera"]["shake_band_hz"], rng)
        if quant_px:
            y = cs.quantize(y, quant_px)
        fref = f0_override if f0_override else m["f0_hz"]
        for s0, w in S.windows(y, fs_out, cfg["signal"]["window_sec"], cfg["signal"]["hop_sec"]):
            f, r, av = F.extract(w, fs_out, fref, cfg)
            X.append(F.vector(f))
            Xr.append(np.array([r[k] for k in F.RAW], dtype=np.float64))
            meta.append(dict(file_id=m["file_id"], label=m["label"], severity=m["severity"],
                             machine_id=m["machine_id"], source=m["source"],
                             f0_hz=m["f0_hz"], start=s0))
        if progress and i % progress == 0:
            print("  build %d/%d  windows=%d" % (i, len(index), len(X)), flush=True)
    if not X:
        return np.zeros((0, len(F.DIMENSIONLESS))), np.zeros((0, len(F.RAW))), []
    return np.vstack(X), np.vstack(Xr), meta


def build_multi(index, cfg, conditions, domain="displacement", seed=None, progress=None):
    seed = cfg["seed"] if seed is None else seed
    acc = {i: ([], [], []) for i in range(len(conditions))}
    for n, rec in enumerate(index):
        sig, fsb, m = signal_for(rec, cfg, domain=domain)
        if sig is None:
            continue
        for ci, c in enumerate(conditions):
            rng = np.random.default_rng(abs(hash((seed, m["file_id"], ci))) % (2 ** 32))
            fs_out = c.get("fs_out") or fsb
            y = cs.decimate(sig, fsb, fs_out, mode=c.get("mode", "ideal"),
                            exposure_sec=c.get("exposure_sec"))
            if domain == "displacement":
                y = cs.to_pixels(y, cfg["optics"]["pixels_per_meter"])
            nt, sd = c.get("noise_type"), c.get("snr_db")
            if nt and sd is not None:
                if nt == "white":
                    y = cs.add_white_noise(y, sd, rng)
                elif nt == "shake":
                    y = cs.add_shake(y, fs_out, sd, cfg["camera"]["shake_band_hz"], rng)
            if c.get("quant_px"):
                y = cs.quantize(y, c["quant_px"])
            X, Xr, M = acc[ci]
            for s0, w in S.windows(y, fs_out, cfg["signal"]["window_sec"], cfg["signal"]["hop_sec"]):
                f, r, av = F.extract(w, fs_out, m["f0_hz"], cfg)
                X.append(F.vector(f))
                Xr.append(np.array([r[k] for k in F.RAW], dtype=np.float64))
                M.append(dict(file_id=m["file_id"], label=m["label"], severity=m["severity"],
                              machine_id=m["machine_id"], source=m["source"],
                              f0_hz=m["f0_hz"], start=s0))
        if progress and n % progress == 0:
            print("  multi %d/%d" % (n, len(index)), flush=True)
    out = {}
    for ci in range(len(conditions)):
        X, Xr, M = acc[ci]
        out[ci] = (np.vstack(X) if X else np.zeros((0, len(F.DIMENSIONLESS))),
                   np.vstack(Xr) if Xr else np.zeros((0, len(F.RAW))), M)
    return out
