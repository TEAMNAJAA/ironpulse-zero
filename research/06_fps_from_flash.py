import sys
import subprocess
import numpy as np
import imageio_ffmpeg
from scipy.signal import butter, filtfilt

if len(sys.argv) < 2:
    raise SystemExit("usage: python 06_fps_from_flash.py <video> [flash_hz]")

video = sys.argv[1]
flash_hz = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
REF = 240.0
W, H = 128, 72

ff = imageio_ffmpeg.get_ffmpeg_exe()
p = subprocess.run([ff, "-v", "error", "-i", video,
                    "-vf", f"scale={W}:{H},format=gray",
                    "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True)
buf = np.frombuffer(p.stdout, dtype=np.uint8)
if buf.size < W * H * 64:
    print(p.stderr.decode()[-600:])
    raise SystemExit("could not decode enough frames")

n = buf.size // (W * H)
b = buf[:n * W * H].reshape(n, W * H).astype(np.float64).mean(axis=1)


def estimate(sig, label):
    if len(sig) < 128:
        print(f"{label:<12} too short")
        return None
    x = sig - sig.mean()
    bb, aa = butter(4, 0.02, btype="high")
    x = filtfilt(bb, aa, x)
    w = np.hanning(len(x))
    S = np.abs(np.fft.rfft(x * w))
    fr = np.fft.rfftfreq(len(x), 1 / REF)
    m = fr > 0.5
    k = int(np.argmax(S[m]))
    peak = fr[m][k]
    lo = max(0, k - 1)
    y0, y1, y2 = S[m][lo], S[m][k], S[m][min(k + 1, len(S[m]) - 1)]
    denom = y0 - 2 * y1 + y2
    if denom != 0:
        peak += 0.5 * (y0 - y2) / denom * (fr[1] - fr[0])
    snr = y1 / np.median(S[m])
    true_fps = REF * flash_hz / peak
    print(f"{label:<12} frames={len(sig):5d}  peak={peak:7.3f}  snr={snr:6.1f}  "
          f"-> true fps = {true_fps:8.2f}")
    return true_fps, snr


print(f"video      = {video}")
print(f"frames     = {n}")
print(f"flash rate = {flash_hz} Hz")
print()
res_all = estimate(b, "whole clip")
a = n // 3
r1 = estimate(b[:a], "first 1/3")
r2 = estimate(b[a:2 * a], "middle 1/3")
r3 = estimate(b[2 * a:], "last 1/3")
print()

if res_all is None or res_all[1] < 8:
    raise SystemExit("FAIL: no clear flash peak. Fill the frame with the flashing screen, "
                     "turn other lights off, and make sure the clip is at least 10 s long.")

vals = [r[0] for r in (r1, r2, r3) if r and r[1] >= 8]
if len(vals) == 3 and (max(vals) - min(vals)) / np.mean(vals) > 0.05:
    print("RAMP DETECTED: the three sections give different frame rates.")
    print("The export baked in a slow-motion ramp, so parts of this file are")
    print("sampled at different rates. Do not use files like this for the dataset.")
else:
    f = res_all[0]
    snap = min([30, 60, 120, 240], key=lambda c: abs(c - f))
    err = abs(snap - f) / snap * 100
    print(f"UNIFORM sampling across the clip")
    print(f"measured = {f:.2f} fps  ->  nearest standard rate = {snap} fps  (off by {err:.2f}%)")
    if err < 3:
        print(f"USE fs = {float(snap)} in every other script")
    else:
        print(f"measurement does not land on a standard rate; re-shoot with a longer, "
              f"steadier clip before trusting it")
