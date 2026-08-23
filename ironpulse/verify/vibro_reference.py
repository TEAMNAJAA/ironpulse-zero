import sys
import cv2
import numpy as np


def run(video, fs, points):
    cap = cv2.VideoCapture(video)
    ok, first = cap.read()
    if not ok:
        raise SystemExit("cannot read video")
    p0 = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    prev = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    lk = dict(winSize=(25, 25), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.001),
              flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS, minEigThreshold=1e-4)
    track = [p0.reshape(-1, 2).copy()]
    lost = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, gray, p0, None, **lk)
        if p1 is None or int(st.sum()) < 2:
            lost = 1
            break
        track.append(p1.reshape(-1, 2).copy())
        prev = gray
        p0 = p1
    cap.release()

    t = np.array(track)
    if len(t) < 64:
        raise SystemExit("tracking failed too early")

    machine = t[:, 0, :]
    reference = t[:, 1, :]
    diff = (machine - machine[0]) - (reference - reference[0])
    out = {}
    for axis, name in [(1, "y"), (0, "x")]:
        sig = diff[:, axis]
        idx = np.arange(len(sig))
        sig = sig - np.polyval(np.polyfit(idx, sig, 2), idx)
        win = np.hanning(len(sig))
        spec = np.abs(np.fft.rfft(sig * win)) * 2 / win.sum()
        freq = np.fft.rfftfreq(len(sig), 1 / fs)
        band = freq > 2
        peak_i = int(np.argmax(spec[band]))
        out[name] = dict(peak_f=float(freq[band][peak_i]),
                         peak_a=float(spec[band][peak_i]),
                         noise=float(np.median(spec[band])),
                         n=len(sig),
                         p2p=float(np.ptp(sig)),
                         freq=freq, spec=spec, sig=sig)
    out["frames"] = len(t)
    out["lost"] = bool(lost)
    out["machine_p2p_y"] = float(np.ptp(machine[:, 1]))
    out["reference_p2p_y"] = float(np.ptp(reference[:, 1]))
    return out


if __name__ == "__main__":
    if len(sys.argv) < 7:
        raise SystemExit("usage: python vibro_reference.py <video> <fs> "
                         "<mx> <my> <rx> <ry>")
    v = sys.argv[1]
    fs = float(sys.argv[2])
    pts = [(float(sys.argv[3]), float(sys.argv[4])),
           (float(sys.argv[5]), float(sys.argv[6]))]
    r = run(v, fs, pts)
    print("frames tracked    = %d   lost=%s" % (r["frames"], r["lost"]))
    for ax in ("y", "x"):
        d = r[ax]
        print("axis %s  PEAK = %.4f Hz  -> %.0f rpm   amp=%.5f px   peak/noise=%.1f"
              % (ax, d["peak_f"], d["peak_f"] * 60, d["peak_a"], d["peak_a"] / d["noise"]))
