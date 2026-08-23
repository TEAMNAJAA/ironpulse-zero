import numpy as np
from scipy import signal as sps

BASE_FS = 48000.0


def effective_highpass(highpass_hz, f0_hz, order_floor):
    fc = float(highpass_hz)
    if f0_hz and order_floor:
        fc = max(fc, float(order_floor) * float(f0_hz))
    return fc


def accel_to_displacement(a, fs, highpass_hz=0.5, hp_order=4, g0=None,
                          f0_hz=None, order_floor=None):
    a = np.asarray(a, dtype=np.float64)
    if g0 is not None:
        a = a * g0
    a = sps.detrend(a, type="linear")
    highpass_hz = effective_highpass(highpass_hz, f0_hz, order_floor)
    n = len(a)
    A = np.fft.rfft(a)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    H = np.zeros_like(f)
    nz = f > 0
    H[nz] = -1.0 / (2.0 * np.pi * f[nz]) ** 2
    hp = np.zeros_like(f)
    hp[nz] = 1.0 / np.sqrt(1.0 + (highpass_hz / f[nz]) ** (2 * hp_order))
    X = A * H * hp
    x = np.fft.irfft(X, n=n)
    return x


def to_pixels(x_meters, pixels_per_meter):
    return np.asarray(x_meters, dtype=np.float64) * pixels_per_meter


def rebase(x, fs_in, fs_base=BASE_FS):
    if abs(fs_in - fs_base) < 1e-9:
        return np.asarray(x, dtype=np.float64), fs_base
    from math import gcd
    up = int(round(fs_base))
    dn = int(round(fs_in))
    g = gcd(up, dn)
    y = sps.resample_poly(np.asarray(x, dtype=np.float64), up // g, dn // g)
    return y, fs_base


def _factor(fs_in, fs_out):
    r = fs_in / float(fs_out)
    n = int(round(r))
    if abs(r - n) > 1e-6:
        raise ValueError("non-integer decimation %s -> %s" % (fs_in, fs_out))
    return max(n, 1)


def decimate_ideal(x, fs_in, fs_out):
    n = _factor(fs_in, fs_out)
    if n == 1:
        return np.asarray(x, dtype=np.float64)
    y = np.asarray(x, dtype=np.float64)
    while n > 1:
        step = min(n, 10)
        while n % step != 0 and step > 1:
            step -= 1
        if step == 1:
            step = n
        y = sps.decimate(y, step, ftype="fir", zero_phase=True)
        n //= step
    return y


def decimate_naive(x, fs_in, fs_out):
    n = _factor(fs_in, fs_out)
    return np.asarray(x, dtype=np.float64)[::n].copy()


def decimate_boxcar(x, fs_in, fs_out, exposure_sec):
    n = _factor(fs_in, fs_out)
    w = int(round(exposure_sec * fs_in))
    w = max(1, min(w, n))
    x = np.asarray(x, dtype=np.float64)
    if w > 1:
        k = np.ones(w) / w
        x = np.convolve(x, k, mode="same")
    return x[::n].copy()


def decimate(x, fs_in, fs_out, mode="ideal", exposure_sec=None):
    if mode == "ideal":
        return decimate_ideal(x, fs_in, fs_out)
    if mode == "naive":
        return decimate_naive(x, fs_in, fs_out)
    if mode == "boxcar":
        if exposure_sec is None:
            exposure_sec = 1.0 / fs_out
        return decimate_boxcar(x, fs_in, fs_out, exposure_sec)
    raise ValueError("unknown mode %s" % mode)


def quantize(x_px, step_px):
    if step_px is None or step_px <= 0:
        return np.asarray(x_px, dtype=np.float64)
    return np.round(np.asarray(x_px, dtype=np.float64) / step_px) * step_px


def _scale_for_snr(sig, noise, snr_db):
    ps = float(np.mean(np.asarray(sig, dtype=np.float64) ** 2))
    pn = float(np.mean(np.asarray(noise, dtype=np.float64) ** 2))
    if pn <= 0 or ps <= 0:
        return 0.0
    target = ps / (10.0 ** (snr_db / 10.0))
    return float(np.sqrt(target / pn))


def add_white_noise(x, snr_db, rng):
    x = np.asarray(x, dtype=np.float64)
    n = rng.standard_normal(len(x))
    return x + _scale_for_snr(x, n, snr_db) * n


def make_shake(nsamp, fs, band, rng):
    n = rng.standard_normal(nsamp)
    lo, hi = band
    hi = min(hi, fs / 2.0 * 0.98)
    if lo >= hi:
        return np.zeros(nsamp)
    b, a = sps.butter(4, [lo / (fs / 2.0), hi / (fs / 2.0)], btype="band")
    return sps.filtfilt(b, a, n)


def add_shake(x, fs, snr_db, band, rng):
    x = np.asarray(x, dtype=np.float64)
    s = make_shake(len(x), fs, band, rng)
    if np.allclose(s, 0):
        return x
    return x + _scale_for_snr(x, s, snr_db) * s


def camera_chain(accel, fs_in, fs_out, mode, cfg, exposure_sec=None,
                 quant_px=None, snr_db=None, noise_type=None, rng=None,
                 already_displacement=False, f0_hz=None):
    if already_displacement:
        d = np.asarray(accel, dtype=np.float64)
    else:
        d = accel_to_displacement(accel, fs_in,
                                  highpass_hz=cfg["integration"]["highpass_hz"],
                                  hp_order=cfg["integration"]["highpass_order"],
                                  g0=cfg["integration"]["g0"],
                                  f0_hz=f0_hz,
                                  order_floor=cfg["integration"].get("highpass_order_floor"))
    d, fs_b = rebase(d, fs_in, BASE_FS)
    y = decimate(d, fs_b, fs_out, mode=mode, exposure_sec=exposure_sec)
    y = to_pixels(y, cfg["optics"]["pixels_per_meter"])
    if snr_db is not None and noise_type:
        rng = rng if rng is not None else np.random.default_rng(cfg["seed"])
        if noise_type == "white":
            y = add_white_noise(y, snr_db, rng)
        elif noise_type == "shake":
            y = add_shake(y, fs_out, snr_db, cfg["camera"]["shake_band_hz"], rng)
    if quant_px:
        y = quantize(y, quant_px)
    return y, fs_out
