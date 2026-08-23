import os, re, io, csv, zipfile
import numpy as np
from . import dsp, config

DS = os.path.join(config.ROOT, "datasets")
CWRU_RAW = os.path.join(DS, "raw", "cwru")
MAF_RAW = os.path.join(DS, "raw", "mafaulda")
MAF_FS = 50000.0
CWRU_PAGE_FS = {"normal-baseline-data": 48000.0,
                "48k-drive-end-bearing-fault-data": 48000.0,
                "12k-drive-end-bearing-fault-data": 12000.0,
                "12k-fan-end-bearing-fault-data": 12000.0}
MAF_CLASS = {"normal": "normal",
             "imbalance": "imbalance",
             "horizontal-misalignment": "horizontal_misalignment",
             "vertical-misalignment": "vertical_misalignment"}
MAF_CHANNELS = {"tach": 0, "under_axial": 1, "under_radial": 2, "under_tang": 3,
                "over_axial": 4, "over_radial": 5, "over_tang": 6, "mic": 7}
PRIMARY_CHANNEL = "under_radial"


def refine_peak(f, A, target, rel_window=0.2):
    lo, hi = target * (1.0 - rel_window), target * (1.0 + rel_window)
    m = (f >= lo) & (f <= hi)
    if not m.any():
        return None
    sub = np.flatnonzero(m)
    j = sub[int(np.argmax(A[m]))]
    if 0 < j < len(A) - 1:
        a, b, c = A[j - 1], A[j], A[j + 1]
        den = a - 2 * b + c
        d = 0.5 * (a - c) / den if abs(den) > 1e-30 else 0.0
        d = float(np.clip(d, -0.5, 0.5))
    else:
        d = 0.0
    df = f[1] - f[0]
    return float(f[j] + d * df)


def f0_from_tacho(tach, fs, nominal, rel_window=0.2):
    f, A = dsp.amplitude_spectrum(tach, fs, mode="constant", win="hann")
    return refine_peak(f, A, nominal, rel_window)


def cwru_index():
    path = os.path.join(DS, "cwru_manifest.csv")
    out = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        p = os.path.join(CWRU_RAW, r["file_id"] + ".mat")
        if not os.path.exists(p):
            continue
        out.append(dict(source="cwru", file_id="cwru_%s_%s" % (r["page"][:3], r["file_id"]),
                        raw_id=r["file_id"], path=p, page=r["page"],
                        fs=CWRU_PAGE_FS[r["page"]], label=r["label"],
                        severity=float(r["severity_in"]) if r["severity_in"] else None,
                        machine_id="cwru_%s" % r["sensor_end"],
                        load_hp=r["load_hp"], rpm_doc=float(r["rpm"]) if r["rpm"] else None,
                        name=r["name"], sensor_end=r["sensor_end"]))
    return out


def cwru_load(rec, channel=None):
    import scipy.io as sio
    m = sio.loadmat(rec["path"])
    end = channel or ("DE" if rec["sensor_end"] in ("DE", "normal") else "FE")
    key = [k for k in m if k.endswith("_%s_time" % end)]
    if not key:
        key = [k for k in m if k.endswith("_DE_time")]
    if not key:
        return None
    x = np.asarray(m[key[0]], dtype=np.float64).ravel()
    rpm = None
    rk = [k for k in m if k.endswith("RPM")]
    if rk:
        try:
            rpm = float(np.asarray(m[rk[0]]).ravel()[0])
        except Exception:
            rpm = None
    if rpm is None:
        rpm = rec["rpm_doc"]
    method = "doc_rpm_matfile" if rk else "doc_rpm_table"
    f0 = rpm / 60.0 if rpm else None
    if f0:
        f, A = dsp.amplitude_spectrum(x, rec["fs"], mode="linear", win="hann")
        r = refine_peak(f, A, f0, rel_window=0.06)
        if r:
            f0, method = r, method + "+spectral_refine"
    return dict(signal=x, fs=rec["fs"], f0_hz=f0, f0_method=method,
                label=rec["label"], severity=rec["severity"],
                machine_id=rec["machine_id"], source="cwru",
                channel=end, file_id=rec["file_id"])


def _maf_label(member):
    parts = member.strip("/").split("/")
    top = parts[0]
    if top in MAF_CLASS:
        lab = MAF_CLASS[top]
        sev = parts[1] if len(parts) > 2 else None
    else:
        sub = parts[1] if len(parts) > 2 else "unknown"
        lab = "%s_%s" % (top, sub)
        sev = parts[2] if len(parts) > 3 else None
    return lab, sev


def maf_index():
    out = []
    for z in sorted(os.listdir(MAF_RAW)) if os.path.isdir(MAF_RAW) else []:
        if not z.endswith(".zip"):
            continue
        p = os.path.join(MAF_RAW, z)
        try:
            zf = zipfile.ZipFile(p)
        except Exception:
            continue
        for n in zf.namelist():
            if not n.endswith(".csv"):
                continue
            lab, sev = _maf_label(n)
            base = os.path.basename(n)[:-4]
            try:
                nominal = float(base)
            except ValueError:
                continue
            out.append(dict(source="mafaulda", archive=p, member=n,
                            file_id="maf_" + n.replace("/", "_")[:-4],
                            fs=MAF_FS, label=lab, severity_raw=sev,
                            nominal_f0=nominal, machine_id="mafaulda_mfs_abvt"))
        zf.close()
    return out


def maf_load(rec, channel=PRIMARY_CHANNEL):
    import pandas as pd
    zf = zipfile.ZipFile(rec["archive"])
    raw = zf.read(rec["member"])
    zf.close()
    a = pd.read_csv(io.BytesIO(raw), header=None, dtype=np.float32).values
    tach = a[:, MAF_CHANNELS["tach"]].astype(np.float64)
    x = a[:, MAF_CHANNELS[channel]].astype(np.float64)
    f0 = f0_from_tacho(tach, rec["fs"], rec["nominal_f0"], rel_window=0.2)
    method = "tachometer_channel"
    if f0 is None:
        f0 = rec["nominal_f0"]
        method = "filename_nominal"
    return dict(signal=x, fs=rec["fs"], f0_hz=f0, f0_method=method,
                label=rec["label"], severity=parse_severity(rec),
                machine_id=rec["machine_id"], source="mafaulda",
                channel=channel, file_id=rec["file_id"])


def parse_severity(rec):
    s = rec.get("severity_raw")
    if not s:
        return None
    m = re.match(r"^([\d.]+)\s*(g|mm)$", s.strip())
    if m:
        return float(m.group(1))
    return None


def windows(x, fs, window_sec, hop_sec):
    n = int(round(window_sec * fs))
    h = int(round(hop_sec * fs))
    if n <= 1 or len(x) < n:
        return
    for s in range(0, len(x) - n + 1, max(h, 1)):
        yield s, x[s:s + n]
