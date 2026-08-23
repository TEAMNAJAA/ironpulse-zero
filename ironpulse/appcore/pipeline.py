import numpy as np

from core import dsp, features as F
from . import video, markers, cfgmap

PEAK_BAND_MIN_HZ = 2.0
MACHINE_AXES = ("d_machine_x", "d_machine_y")


class AnalysisError(Exception):
    pass


def machine_axes(sigs):
    return [n for n in MACHINE_AXES if n in sigs]


def blade_pass_f0(path, box, cfg, fs, blade_count, info=None):
    info = info or video.probe(path)
    vals = []
    for g in video.frames(path, crop=list(box), width=info["width"],
                          height=info["height"]):
        vals.append(float(g.mean()))
    s = np.asarray(vals, dtype=np.float64)
    f, A = dsp.amplitude_spectrum(s, fs, mode=int(cfg["analysis"]["detrend_order"]),
                                  win=cfg["analysis"]["window_fn"])
    m = f > PEAK_BAND_MIN_HZ
    if not m.any():
        raise AnalysisError("สัญญาณความสว่างสั้นเกินไปสำหรับการหาความถี่ใบพัดผ่าน")
    bpf = float(f[m][int(np.argmax(A[m]))])
    return bpf / float(blade_count), bpf


def spectral_f0(sig, fs, cfg, lo, hi):
    f, A = dsp.amplitude_spectrum(sig, fs, mode=int(cfg["analysis"]["detrend_order"]),
                                  win=cfg["analysis"]["window_fn"])
    m = (f >= lo) & (f <= hi)
    if not m.any():
        raise AnalysisError("ไม่มีข้อมูลในช่วงความถี่ที่ระบุ")
    return float(f[m][int(np.argmax(A[m]))])


def band_peak(f, A, lo, hi, select=None, exclude_bins=3):
    m = (f > lo) & (f <= hi)
    if not m.any():
        return None
    fb, Ab = f[m], A[m]
    if select is None:
        i = int(np.argmax(Ab))
    else:
        s = np.nonzero((fb >= select[0]) & (fb <= select[1]))[0]
        if not len(s):
            return None
        i = int(s[int(np.argmax(Ab[s]))])
    a, b = max(0, i - exclude_bins), min(len(Ab), i + exclude_bins + 1)
    rest = np.concatenate([Ab[:a], Ab[b:]])
    if len(rest) and rest.max() > 0:
        prominence = float(Ab[i] / rest.max())
        runner = float(rest.max())
    else:
        prominence = float("inf")
        runner = 0.0
    return dict(freq=float(fb[i]), amp=float(Ab[i]), prominence=prominence,
                runner_up_amp=runner)


def peak_of(sig, fs, cfg, band_min=PEAK_BAND_MIN_HZ, exclude_bins=3):
    f, A = dsp.amplitude_spectrum(sig, fs, mode=int(cfg["analysis"]["detrend_order"]),
                                  win=cfg["analysis"]["window_fn"])
    p = band_peak(f, A, band_min, float(f[-1]), exclude_bins=exclude_bins)
    if p is None:
        raise AnalysisError("สัญญาณสั้นเกินไปสำหรับการหาพีค")
    m = f > band_min
    p = dict(p)
    p["noise"] = float(np.median(A[m]))
    p["f"] = f
    p["A"] = A
    return p


def peak_at_order(sig, fs, cfg, f0, order=1.0, exclude_bins=3,
                  band_min=PEAK_BAND_MIN_HZ):
    f, A = dsp.amplitude_spectrum(sig, fs, mode=int(cfg["analysis"]["detrend_order"]),
                                  win=cfg["analysis"]["window_fn"])
    tol = float(cfg["analysis"]["order_tolerance"])
    lo = float(f0) * (float(order) - tol)
    hi = float(f0) * (float(order) + tol)
    sel = np.nonzero((f >= lo) & (f <= hi))[0]
    if not len(sel):
        return None
    i = int(sel[int(np.argmax(A[sel]))])
    a, b = max(0, i - exclude_bins), min(len(A), i + exclude_bins + 1)
    keep = f > band_min
    keep[a:b] = False
    rest = A[keep]
    if len(rest) and rest.max() > 0:
        prominence = float(A[i] / rest.max())
        runner = float(rest.max())
    else:
        prominence = float("inf")
        runner = 0.0
    return dict(freq=float(f[i]), amp=float(A[i]), prominence=prominence,
                runner_up_amp=runner, noise=float(np.median(A[f > band_min])),
                f=f, A=A)


def primary_axis(sigs, fs, cfg, f0=None):
    best = None
    for axis in machine_axes(sigs):
        pk = peak_at_order(sigs[axis], fs, cfg, f0) if f0 else None
        if pk is None:
            pk = peak_of(sigs[axis], fs, cfg)
        if best is None or pk["prominence"] > best[1]["prominence"]:
            best = (axis, pk)
    if best is None:
        raise AnalysisError(
            "ไม่มีสัญญาณของจุดบนเครื่องให้ตรวจคุณภาพ ต้องมี d_machine_x หรือ d_machine_y")
    return best


def f0_checks(f0, fs, cfg, baseline_f0=None):
    warnings = []
    if f0 > fs / 2.0:
        raise AnalysisError(
            "ความเร็วรอบ %.2f Hz เกินครึ่งหนึ่งของอัตราเฟรม (%.1f Hz) "
            "กล้องสุ่มภาพไม่ทันการหมุน วิเคราะห์ต่อไม่ได้" % (f0, fs / 2.0))
    if f0 * 3.0 > fs / 2.0:
        warnings.append(
            "ฮาร์มอนิกที่ 3 (%.1f Hz) เกินขีดจำกัดการสุ่มตัวอย่างที่ %.1f Hz "
            "จะวินิจฉัยการเยื้องศูนย์จาก 3x ไม่ได้ ต้องลดความเร็วรอบให้ต่ำกว่า %.0f rpm"
            % (f0 * 3.0, fs / 2.0, (fs / 2.0) / 3.0 * 60.0))
    if baseline_f0:
        shift = abs(f0 - baseline_f0) / baseline_f0
        if shift > float(cfg["analysis"]["f0_shift_warn"]):
            warnings.append(
                "ความเร็วรอบเปลี่ยนไป %.1f%% จากตอนสร้าง baseline (%.2f Hz เทียบ %.2f Hz) "
                "ค่าทุกตัวจะขยับพร้อมกันและดูเหมือนความผิดปกติทั้งที่อาจไม่ใช่"
                % (100 * shift, f0, baseline_f0))
    return warnings


def clip_length_checks(n_frames, fs, f0, cfg):
    warnings = []
    revs = n_frames / float(fs) * f0
    need = float(cfg["analysis"]["min_revolutions"])
    if revs < need:
        warnings.append(
            "คลิปยาว %.1f รอบการหมุน ต่ำกว่าขั้นต่ำ %.1f รอบ "
            "ที่ %.2f Hz ควรยาวอย่างน้อย %.1f วินาที"
            % (revs, need, f0, need / f0))
    return warnings


def quality_check(peak, cfg, nyq_order, f0, fs, axis_name=""):
    noise = peak["noise"]
    a1 = peak["amp"]
    ratio = a1 / noise if noise > 0 else float("inf")
    need = float(cfg["quality"]["min_a1_over_noise"])
    warnings = []
    if ratio < need:
        warnings.append(
            "สัญญาณที่ 1x บนแกน %s สูงกว่าพื้นสัญญาณรบกวนแค่ %.1f เท่า "
            "ต้องการอย่างน้อย %.1f เท่า "
            "ผลการวัดครั้งนี้เชื่อถือได้น้อย ให้ขยับกล้องเข้าใกล้ เพิ่มแสง หรือใช้มาร์กเกอร์ใหญ่ขึ้น"
            % (axis_name or "ที่เลือก", ratio, need))
    return warnings, ratio


def prominence_check(peak, cfg, axis_name=""):
    need = float(cfg["quality"]["min_peak_prominence"])
    if peak["prominence"] < need:
        return ["พีคที่พบบนแกน %s สูงกว่าพีครองเพียง %.2f เท่า (ต้องการอย่างน้อย %.2f เท่า) "
                "สเปกตรัมไม่มีพีคที่ชัดเจน ค่าความถี่ที่ได้จึงไม่น่าเชื่อถือ "
                "แกนนี้เป็นแกนที่เด่นที่สุดแล้วในบรรดาแกนที่มี "
                "จึงแปลว่าสัญญาณจมอยู่ในสัญญาณรบกวนทุกแกน "
                "ให้ตรวจสอบว่ากล้องจับเครื่องที่กำลังหมุนอยู่จริง "
                "มาร์กเกอร์ติดแน่นกับตัวเครื่อง และเครื่องเปิดอยู่ตลอดคลิป"
                % (axis_name or "ที่เลือก", peak["prominence"], need)]
    return []


def auto_f0(sigs, fs, cfg, f0_range=None, baseline_f0=None):
    lo = f0_range[0] if f0_range else PEAK_BAND_MIN_HZ
    hi = f0_range[1] if f0_range else fs / 2.0
    tol = float(cfg["analysis"]["f0_shift_warn"])
    need = float(cfg["quality"]["min_peak_prominence"])
    select = None
    if baseline_f0:
        select = (float(baseline_f0) * (1.0 - tol), float(baseline_f0) * (1.0 + tol))
    best_sel, best_wide = None, None
    for axis in machine_axes(sigs):
        f, A = dsp.amplitude_spectrum(sigs[axis], fs,
                                      mode=int(cfg["analysis"]["detrend_order"]),
                                      win=cfg["analysis"]["window_fn"])
        wide = band_peak(f, A, lo, hi)
        if wide and (best_wide is None or wide["prominence"] > best_wide[0]["prominence"]):
            best_wide = (wide, axis)
        if select:
            near = band_peak(f, A, lo, hi, select=select)
            if near and (best_sel is None or near["prominence"] > best_sel[0]["prominence"]):
                best_sel = (near, axis)
    if best_sel and best_sel[0]["prominence"] >= need:
        p, axis = best_sel
        return float(p["freq"]), axis.split("_")[-1], float(p["prominence"]), True
    if best_wide is None:
        raise AnalysisError("หาความเร็วรอบอัตโนมัติไม่ได้ ให้กรอกรอบหมุนหรือจำนวนใบพัด")
    p, axis = best_wide
    return float(p["freq"]), axis.split("_")[-1], float(p["prominence"]), False


def signal_checks(sigs, app_cfg, blade_count, rbp_blades):
    warnings = []
    wanted = list(app_cfg["features"]["signals"])
    missing = [n for n in wanted if n not in sigs]
    if missing:
        warnings.append(
            "config ตั้งให้ใช้สัญญาณ %s แต่การถ่ายครั้งนี้ไม่มี %s "
            "เพราะติดมาร์กเกอร์แค่ %d จุด "
            "feature ของสัญญาณที่ขาดจะไม่ถูกคำนวณและ feature vector จะสั้นลง "
            "baseline ที่สร้างจากจำนวนมาร์กเกอร์ต่างกันเทียบกันไม่ได้"
            % (", ".join(wanted), ", ".join(missing),
               2 if "d_base_y" in missing else 3))
    model_missing = [n for n in app_cfg["features"]["model_signals"] if n not in sigs]
    if model_missing:
        warnings.append(
            "สัญญาณที่โมเดลต้องใช้ขาดไป %s "
            "ห้ามนำ feature vector นี้ไปเทียบกับ baseline ที่มีสัญญาณครบ"
            % ", ".join(model_missing))
    if not blade_count:
        warnings.append(
            "ไม่ได้ระบุจำนวนใบพัด feature rbp จะเป็น 0.0 ทุกครั้ง "
            "ซึ่งแปลว่า ไม่ได้วัด ไม่ใช่ ไม่มีความเปลี่ยนแปลง "
            "ห้ามนำ rbp ไปตีความหรือคำนวณขนาดผลจนกว่าจะกรอกจำนวนใบพัด")
    elif rbp_blades is None:
        need = int(app_cfg["machine"]["rbp_min_blade_count"])
        warnings.append(
            "ใบพัด %d ใบ ทำให้ order ของ blade pass ตรงกับฮาร์มอนิกที่ %d ของเพลาพอดี "
            "ซึ่งคือ feature r%d ที่วัดอยู่แล้ว rbp จึงเท่ากับ r%d ทุกตัวเลขและไม่ให้ข้อมูลเพิ่ม "
            "ทั้งยังแยกใบพัดเสียหายออกจากการเยื้องศูนย์ไม่ได้ "
            "ระบบจึงตั้ง rbp เป็น ไม่ได้วัด และปิดกฎใบพัดเสียหาย "
            "ต้องมีตั้งแต่ %d ใบขึ้นไปจึงจะคำนวณ rbp"
            % (int(blade_count), int(blade_count), int(blade_count),
               int(blade_count), need))
    return warnings


def analyse(path, rois, app_cfg, fs=None, f0_hz=None, blade_count=None,
            blade_box=None, baseline_f0=None, progress=None, f0_range=None):
    fs = float(fs or app_cfg["capture"]["default_fps"])
    detrend_order = int(app_cfg["analysis"]["detrend_order"])
    info = video.probe(path)
    warnings = list(video.check_usable(info, app_cfg))
    t, crop = markers.track(path, rois, app_cfg, info=info, progress=progress)
    warnings += markers.reference_check(t, detrend_order)
    sigs = markers.differential(t)
    rbp_blades = cfgmap.rbp_blade_count(app_cfg, blade_count)
    warnings += signal_checks(sigs, app_cfg, blade_count, rbp_blades)
    n = t.shape[0]
    f0_source = "user"
    f0_locked = None
    if f0_hz is None and blade_box is not None and blade_count:
        f0_hz, bpf = blade_pass_f0(path, blade_box, app_cfg, fs, blade_count, info)
        f0_source = "blade_pass"
    if f0_hz is None:
        f0_hz, f0_axis, f0_prom, f0_locked = auto_f0(sigs, fs, app_cfg, f0_range,
                                                     baseline_f0)
        f0_source = "spectral_peak_%s" % f0_axis
        warnings.append(
            "ไม่ได้ระบุความเร็วรอบ ระบบเดาจากพีคที่เด่นที่สุดบนแกน %s ได้ %.3f Hz (%.0f rpm) "
            "เด่นกว่าพีครอง %.2f เท่า ถ้าทราบรอบจริงหรือจำนวนใบพัด ควรกรอกเพื่อความแม่นยำ"
            % (f0_axis, f0_hz, f0_hz * 60, f0_prom))
        if baseline_f0 and not f0_locked:
            warnings.append(
                "หาพีคความเร็วรอบใกล้ค่าของ baseline (%.3f Hz ภายใน %.0f%%) ไม่เจอ "
                "ระบบจึงถอยไปใช้พีคที่เด่นที่สุดในทั้งย่านแทนและได้ %.3f Hz "
                "ถ้าเครื่องหมุนเท่าเดิม แปลว่าคลิปนี้สัญญาณ 1x อ่อนจนพีคหลอกชนะ"
                % (baseline_f0, 100 * float(app_cfg["analysis"]["f0_shift_warn"]), f0_hz))
    f0_hz = float(f0_hz)
    warnings += f0_checks(f0_hz, fs, app_cfg, baseline_f0)
    warnings += clip_length_checks(n, fs, f0_hz, app_cfg)
    ccfg = cfgmap.core_cfg(app_cfg, fs, f0_hz, rbp_blades)
    nyq_order = ccfg["features"]["max_order"]
    out_sigs, out_feats, out_raw, out_avail, spectra = {}, {}, {}, {}, {}
    for name in app_cfg["features"]["signals"]:
        if name not in sigs:
            continue
        sig = sigs[name]
        feat, raw, avail = F.extract(sig, fs, f0_hz, ccfg)
        out_sigs[name] = sig
        out_feats[name] = feat
        out_raw[name] = raw
        out_avail[name] = avail
        pk = peak_of(sig, fs, app_cfg)
        spectra[name] = dict(freq=pk["f"], amp=pk["A"], peak_hz=pk["freq"],
                             peak_amp=pk["amp"], noise=pk["noise"],
                             prominence=pk["prominence"])
    primary_name, primary = primary_axis(sigs, fs, app_cfg, f0_hz)
    qw, ratio = quality_check(primary, app_cfg, nyq_order, f0_hz, fs, primary_name)
    warnings += qw
    warnings += prominence_check(primary, app_cfg, primary_name)
    return dict(info=info, crop=crop, track=t, signals=out_sigs, features=out_feats,
                raw=out_raw, available=out_avail, spectra=spectra, fs=fs, f0_hz=f0_hz,
                f0_source=f0_source, f0_locked_to_baseline=f0_locked,
                n_frames=n, nyq_order=nyq_order, peak=primary,
                primary_axis=primary_name, blade_count=blade_count,
                rbp_blade_count=rbp_blades,
                signals_missing=[n for n in app_cfg["features"]["signals"]
                                 if n not in sigs],
                a1_over_noise=ratio, warnings=warnings,
                feature_vector=feature_vector(out_feats, app_cfg))


def feature_vector(out_feats, app_cfg):
    keep = app_cfg["features"]["keep"]
    names, vals = [], []
    for sname in app_cfg["features"]["model_signals"]:
        if sname not in out_feats:
            continue
        for k in keep:
            names.append("%s.%s" % (sname, k))
            vals.append(out_feats[sname][k])
    return np.array(vals, dtype=np.float64), names
