import os, sys, csv
import numpy as np
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.environ.setdefault("IRONPULSE_PROJECT_ROOT", os.path.join(_REPO, "research"))
from core import camera_sim as cs, dsp, detectors, config

CFG = config.CFG
G0 = CFG["integration"]["g0"]
PPM = CFG["optics"]["pixels_per_meter"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "selftest_results.csv")

rows = []


def rec(name, measured, expected, unit, tol, note=""):
    err = abs(measured - expected)
    ok = err <= tol
    rows.append(dict(check=name, measured=round(measured, 6), expected=round(expected, 6),
                     unit=unit, abs_error=round(err, 8), tolerance=tol,
                     result="PASS" if ok else "FAIL", note=note))
    print("%-46s %12.5f vs %12.5f %-4s  %s" % (name, measured, expected, unit,
                                               "PASS" if ok else "FAIL"))
    return ok


def sine_accel_g(amp_m, f0, fs, dur):
    t = np.arange(0, dur, 1.0 / fs)
    return -amp_m * (2 * np.pi * f0) ** 2 * np.sin(2 * np.pi * f0 * t) / G0, t


def check_integration():
    fs, f0, amp = 48000.0, 30.0, 50e-6
    a, _ = sine_accel_g(amp, f0, fs, 2.0)
    d = cs.accel_to_displacement(a, fs, highpass_hz=CFG["integration"]["highpass_hz"],
                                 hp_order=CFG["integration"]["highpass_order"], g0=G0,
                                 f0_hz=f0,
                                 order_floor=CFG["integration"]["highpass_order_floor"])
    f, A = dsp.amplitude_spectrum(d, fs)
    got = A[np.argmax(A)] * 1e6
    rec("double integration recovers reference", got, 50.0, "um", 0.05,
        "acceleration -> displacement, spec 6.2.1")


def check_highpass_literal():
    fs, f0, amp = 48000.0, 30.0, 50e-6
    a, _ = sine_accel_g(amp, f0, fs, 2.0)
    rng = np.random.default_rng(CFG["seed"])
    a = a + 0.02 * rng.standard_normal(len(a)) / G0
    d = cs.accel_to_displacement(a, fs, highpass_hz=0.5, hp_order=4, g0=G0,
                                 f0_hz=f0, order_floor=None)
    f, A = dsp.amplitude_spectrum(d, fs)
    peak_f = f[np.argmax(A)]
    peak_um = A.max() * 1e6
    rows.append(dict(check="literal 0.5 Hz high-pass leaves a sub-Hz artifact",
                     measured=round(peak_um, 3), expected=50.0, unit="um",
                     abs_error=round(abs(peak_um - 50.0), 3), tolerance="n/a",
                     result="EXPECTED FAILURE",
                     note="artifact peak at %.2f Hz; this is why Q1 raises the cutoff" % peak_f))
    print("%-46s %12.3f vs %12.3f um   EXPECTED FAILURE (peak at %.2f Hz)"
          % ("literal 0.5 Hz high-pass artifact", peak_um, 50.0, peak_f))


def check_boxcar_sinc():
    fs, f0, amp, fps = 48000.0, 30.0, 50e-6, 240.0
    a, _ = sine_accel_g(amp, f0, fs, 2.0)
    d = cs.accel_to_displacement(a, fs, highpass_hz=CFG["integration"]["highpass_hz"],
                                 hp_order=CFG["integration"]["highpass_order"], g0=G0,
                                 f0_hz=f0,
                                 order_floor=CFG["integration"]["highpass_order_floor"])
    for exposure in [1 / 240.0, 1 / 500.0, 1 / 1000.0]:
        y = cs.decimate(d, fs, fps, mode="boxcar", exposure_sec=exposure)
        f, A = dsp.amplitude_spectrum(y, fps)
        got = A[np.argmax(A)] * 1e6
        w = max(1, min(int(round(exposure * fs)), int(round(fs / fps))))
        eff = w / fs
        x = np.pi * f0 * eff
        theory = 50.0 * abs(np.sin(x) / x)
        rec("boxcar shutter 1/%d s vs sinc theory" % round(1 / exposure), got, theory,
            "um", 0.02, "50 * sinc(f0 * T_exposure)")


def check_naive_and_ideal():
    fs, f0, amp, fps = 48000.0, 30.0, 50e-6, 240.0
    a, _ = sine_accel_g(amp, f0, fs, 2.0)
    d = cs.accel_to_displacement(a, fs, highpass_hz=CFG["integration"]["highpass_hz"],
                                 hp_order=CFG["integration"]["highpass_order"], g0=G0,
                                 f0_hz=f0,
                                 order_floor=CFG["integration"]["highpass_order_floor"])
    for mode, tol in [("naive", 0.02), ("ideal", 0.30)]:
        y = cs.decimate(d, fs, fps, mode=mode)
        f, A = dsp.amplitude_spectrum(y, fps)
        rec("%s decimation preserves amplitude" % mode, A.max() * 1e6, 50.0, "um", tol,
            "no attenuation expected below Nyquist")


def check_aliasing():
    fs, fps = 48000.0, 240.0
    f_true = 300.0
    t = np.arange(0, 2.0, 1.0 / fs)
    x = np.sin(2 * np.pi * f_true * t)
    y = cs.decimate(x, fs, fps, mode="naive")
    f, A = dsp.amplitude_spectrum(y, fps)
    got = f[np.argmax(A)]
    rec("naive sampling folds 300 Hz to its alias", got, abs(f_true - fps), "Hz", 1.0,
        "300 Hz at 240 fps aliases to 60 Hz")


def check_quantisation_noise():
    rng = np.random.default_rng(CFG["seed"])
    x = rng.standard_normal(400000)
    q = 0.05
    e = cs.quantize(x, q) - x
    rec("quantisation noise rms equals q/sqrt(12)", float(e.std()), q / np.sqrt(12),
        "px", 0.0005, "uniform quantiser, spec 6.2.3")


def check_snr_scaling():
    rng = np.random.default_rng(CFG["seed"])
    x = rng.standard_normal(200000)
    for snr in [20.0, 10.0, 0.0]:
        y = cs.add_white_noise(x, snr, np.random.default_rng(1))
        n = y - x
        got = 10 * np.log10(np.mean(x ** 2) / np.mean(n ** 2))
        rec("white noise injected at %.0f dB SNR" % snr, float(got), snr, "dB", 0.15)


def check_pixel_scale():
    rec("50 um converts to pixels", 50e-6 * PPM, 0.145, "px", 0.0005,
        "29 px/cm at 30 cm on 720p")
    rec("one pixel in micrometres", 1e6 / PPM, 344.83, "um", 0.05)


def check_detectors_separable():
    rng = np.random.default_rng(CFG["seed"])
    Xn = rng.standard_normal((400, 13))
    Xa = rng.standard_normal((100, 13)) + 2.5
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.zeros(100), np.ones(100)]
    for cls in detectors.ALL:
        det = cls(CFG).fit(Xn)
        s = np.r_[det.score(Xn[:100]), det.score(Xa)]
        rec("detector %s separates synthetic data" % cls.name,
            float(roc_auc_score(y, s)), 1.0, "auc", 0.02,
            "shifted Gaussian, code check only")


if __name__ == "__main__":
    print("SYNTHETIC SELF-TEST - code validation only, never reported as a research result")
    print("=" * 96)
    check_integration()
    check_highpass_literal()
    check_boxcar_sinc()
    check_naive_and_ideal()
    check_aliasing()
    check_quantisation_noise()
    check_snr_scaling()
    check_pixel_scale()
    check_detectors_separable()
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n_fail = len([r for r in rows if r["result"] == "FAIL"])
    print("=" * 96)
    print("%d checks, %d FAIL, 1 expected failure (the literal 0.5 Hz high-pass)" %
          (len(rows), n_fail))
    print("wrote", OUT)
    sys.exit(1 if n_fail else 0)
