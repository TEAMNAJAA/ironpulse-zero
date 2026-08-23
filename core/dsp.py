import numpy as np
from scipy import signal as sps
from scipy.stats import kurtosis


def poly_detrend(x, order):
    n = len(x)
    if n < order + 2:
        return x - x.mean()
    t = np.arange(n, dtype=np.float64)
    return x - np.polyval(np.polyfit(t, x, int(order)), t)


def detrend_window(x, mode="linear", win="hann"):
    x = np.asarray(x, dtype=np.float64)
    if isinstance(mode, str) and mode.startswith("poly"):
        mode = int(mode[4:])
    if isinstance(mode, (int, np.integer)):
        x = poly_detrend(x, mode)
    elif mode == "linear":
        x = sps.detrend(x, type="linear")
    elif mode == "constant":
        x = x - x.mean()
    if win == "hann":
        w = np.hanning(len(x))
    elif win == "hamming":
        w = np.hamming(len(x))
    else:
        w = np.ones(len(x))
    return x * w, w


def amplitude_spectrum(x, fs, mode="linear", win="hann"):
    xw, w = detrend_window(x, mode, win)
    n = len(xw)
    sc = 2.0 / np.sum(w)
    X = np.fft.rfft(xw)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    A = np.abs(X) * sc
    if len(A):
        A[0] *= 0.5
        if n % 2 == 0:
            A[-1] *= 0.5
    return f, A


def to_order_axis(f, A, f0):
    if f0 is None or f0 <= 0:
        return None, None
    return f / f0, A


def resample_to_order_grid(orders, A, grid):
    out = np.zeros_like(grid)
    valid = grid <= orders[-1]
    if valid.any():
        out[valid] = np.interp(grid[valid], orders, A)
    return out, valid


def peak_amplitude_at_order(orders, A, k, halfwidth):
    lo, hi = k - halfwidth, k + halfwidth
    m = (orders >= lo) & (orders <= hi)
    if not m.any():
        return 0.0, False
    return float(A[m].max()), True


def band_energy(orders, A, lo, hi):
    m = (orders >= lo) & (orders < hi)
    if not m.any():
        return 0.0, False
    return float(np.sum(A[m] ** 2)), True


def spectral_centroid_order(orders, A, lo=0.1, hi=None):
    hi = hi if hi is not None else orders[-1]
    m = (orders >= lo) & (orders <= hi)
    if not m.any():
        return 0.0
    p = A[m] ** 2
    s = p.sum()
    if s <= 0:
        return 0.0
    return float(np.sum(orders[m] * p) / s)


def time_stats(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    crest = peak / rms if rms > 0 else 0.0
    k = float(kurtosis(x, fisher=False)) if rms > 0 else 0.0
    return dict(rms=rms, peak=peak, crest=crest, kurt=k)


def estimate_f0_spectral(x, fs, lo=5.0, hi=100.0, mode="linear"):
    f, A = amplitude_spectrum(x, fs, mode=mode)
    m = (f >= lo) & (f <= min(hi, fs / 2.0 - 1e-9))
    if not m.any():
        return None
    idx = np.argmax(A[m])
    return float(f[m][idx])


def estimate_f0_tacho(tach, fs, lo=5.0, hi=100.0):
    t = np.asarray(tach, dtype=np.float64)
    t = t - t.mean()
    if np.allclose(t, 0):
        return None
    return estimate_f0_spectral(t, fs, lo, hi)
