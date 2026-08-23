import numpy as np
import cv2

from core import dsp
from . import video


class TrackingError(Exception):
    pass


def roi_from_point(cx, cy, margin, width, height):
    x0 = int(max(0, round(cx - margin)))
    y0 = int(max(0, round(cy - margin)))
    x1 = int(min(width, round(cx + margin)))
    y1 = int(min(height, round(cy + margin)))
    return [x0, y0, x1 - x0, y1 - y0]


def pyramid_safe_margin(cfg):
    win = int(cfg["tracking"]["win_size"])
    lvl = int(cfg["tracking"]["max_level"])
    return (win // 2) * (2 ** lvl)


def crop_margin(cfg):
    return max(int(cfg["tracking"]["roi_margin_px"]), pyramid_safe_margin(cfg))


def bounding_crop(rois, margin, width, height, align=2):
    xs0 = min(r[0] for r in rois)
    ys0 = min(r[1] for r in rois)
    xs1 = max(r[0] + r[2] for r in rois)
    ys1 = max(r[1] + r[3] for r in rois)
    x0 = int(max(0, xs0 - margin))
    y0 = int(max(0, ys0 - margin))
    x1 = int(min(width, xs1 + margin))
    y1 = int(min(height, ys1 + margin))
    x0 -= x0 % align
    y0 -= y0 % align
    w = x1 - x0
    h = y1 - y0
    w -= w % align
    h -= h % align
    return [x0, y0, w, h]


def find_in_roi(gray, roi, cfg, index=None):
    x, y, w, h = roi
    patch = gray[y:y + h, x:x + w]
    if patch.size == 0:
        raise TrackingError("กรอบมาร์กเกอร์อยู่นอกภาพ")
    corners = cv2.goodFeaturesToTrack(patch, maxCorners=8, qualityLevel=0.01,
                                      minDistance=5, blockSize=7)
    if corners is None:
        raise TrackingError(
            "หามุมเด่นในกรอบมาร์กเกอร์จุดที่ %s ไม่เจอ "
            "ให้คลิกเลือกตำแหน่งเอง หรือเพิ่มแสง หรือใช้มาร์กเกอร์ใหญ่ขึ้น"
            % ("?" if index is None else index + 1))
    cxy = np.array([w * 0.5, h * 0.5], dtype=np.float32)
    d = np.linalg.norm(corners.reshape(-1, 2) - cxy, axis=1)
    corners = corners[int(np.argmin(d))].reshape(1, 1, 2)
    win = int(cfg["tracking"]["subpix_win"])
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            int(cfg["tracking"]["max_iter"]), float(cfg["tracking"]["epsilon"]))
    cv2.cornerSubPix(patch, corners, (win, win), (-1, -1), crit)
    return float(corners[0, 0, 0] + x), float(corners[0, 0, 1] + y)


def checker_kernel(half):
    k = np.zeros((2 * half, 2 * half), np.float32)
    k[:half, :half] = 1.0
    k[half:, half:] = 1.0
    k[:half, half:] = -1.0
    k[half:, :half] = -1.0
    return k


def reacquire(gray, point, cfg):
    half = int(cfg["tracking"]["checker_half_px"])
    rad = int(cfg["tracking"]["reacquire_radius_px"])
    need = float(cfg["tracking"]["reacquire_min_response"])
    g = gray.astype(np.float32)
    h, w = g.shape
    cx, cy = int(round(point[0])), int(round(point[1]))
    y0, y1 = max(0, cy - rad), min(h, cy + rad)
    x0, x1 = max(0, cx - rad), min(w, cx + rad)
    if y1 - y0 < 2 * half or x1 - x0 < 2 * half:
        return (float(cx), float(cy)), 0.0
    sub = cv2.GaussianBlur(g[y0:y1, x0:x1], (0, 0), 1.0)
    r = cv2.filter2D(sub, cv2.CV_32F, checker_kernel(half))
    i = np.unravel_index(int(np.argmax(np.abs(r))), r.shape)
    strength = float(abs(r[i]))
    if strength < need:
        return (float(cx), float(cy)), strength
    return (float(x0 + i[1]), float(y0 + i[0])), strength


def reacquire_all(gray, points, cfg):
    out, weak = [], []
    for i, p in enumerate(points):
        q, strength = reacquire(gray, p, cfg)
        out.append(q)
        if strength < float(cfg["tracking"]["reacquire_min_response"]):
            weak.append((i + 1, strength, float(np.hypot(q[0] - p[0], q[1] - p[1]))))
    return out, weak


def seed_points(gray, rois, cfg):
    return np.array([find_in_roi(gray, r, cfg, i) for i, r in enumerate(rois)],
                    dtype=np.float32)


def track(path, rois, cfg, info=None, progress=None):
    info = info or video.probe(path)
    W, H = info["width"], info["height"]
    crop = bounding_crop(rois, crop_margin(cfg), W, H)
    cx, cy = crop[0], crop[1]
    local_rois = [[r[0] - cx, r[1] - cy, r[2], r[3]] for r in rois]
    lk = dict(winSize=(int(cfg["tracking"]["win_size"]),
                       int(cfg["tracking"]["win_size"])),
              maxLevel=int(cfg["tracking"]["max_level"]),
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        int(cfg["tracking"]["max_iter"]),
                        float(cfg["tracking"]["epsilon"])),
              flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
              minEigThreshold=float(cfg["tracking"]["min_eig_threshold"]))
    prev = None
    p0 = None
    track_xy = []
    n = 0
    limit = video.frames_used(info, cfg)
    for gray in video.frames(path, crop=crop, width=W, height=H, limit=limit):
        if prev is None:
            p0 = seed_points(gray, local_rois, cfg).reshape(-1, 1, 2)
            track_xy.append(p0.reshape(-1, 2).copy())
            prev = gray.copy()
            n = 1
            continue
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev, gray, p0, None, **lk)
        if p1 is None or st is None or int(st.sum()) < len(rois):
            raise TrackingError(
                "การติดตามหลุดที่เฟรมที่ %d จาก %d "
                "สาเหตุที่พบบ่อยคือแสงน้อยเกินไป มาร์กเกอร์เล็กเกินไป หรือกล้องขยับ"
                % (n, limit))
        track_xy.append(p1.reshape(-1, 2).copy())
        prev = gray
        p0 = p1
        n += 1
        if progress:
            if callable(progress):
                progress(n, limit)
            elif n % progress == 0:
                print("  tracked %d/%d" % (n, info["n_frames"]), flush=True)
    if len(track_xy) < 2:
        raise TrackingError("อ่านเฟรมได้ไม่พอสำหรับการติดตาม")
    t = np.array(track_xy, dtype=np.float64)
    t[:, :, 0] += cx
    t[:, :, 1] += cy
    return t, crop


def ref_index(t):
    return 2 if t.shape[1] > 2 else 1


def has_base(t):
    return t.shape[1] > 2


def differential(t):
    machine = t[:, 0, :]
    j = ref_index(t)
    ref = t[:, j, :]
    base = t[:, 1, :] if has_base(t) else None
    d_machine = (machine - machine[0]) - (ref - ref[0])
    out = {"d_machine_x": d_machine[:, 0], "d_machine_y": d_machine[:, 1]}
    if base is not None:
        d_base = (base - base[0]) - (ref - ref[0])
        out["d_base_x"] = d_base[:, 0]
        out["d_base_y"] = d_base[:, 1]
    return out


def motion_summary(t, index, detrend_order):
    p2p = float(np.hypot(*np.ptp(t[:, index, :], axis=0)))
    rms = float(np.hypot(*[dsp.poly_detrend(t[:, index, k], detrend_order).std()
                           for k in (0, 1)]))
    return p2p, rms


def reference_check(t, detrend_order=2):
    warnings = []
    if t.shape[1] < 2:
        warnings.append(
            "มีมาร์กเกอร์จุดเดียว ตรวจสอบจุดอ้างอิงไม่ได้ "
            "ต้องมีอย่างน้อย 2 จุดคือจุดบนเครื่องกับจุดอ้างอิง")
        return warnings
    j = ref_index(t)
    p_mach, r_mach = motion_summary(t, 0, detrend_order)
    p_ref, r_ref = motion_summary(t, j, detrend_order)
    if p_ref > p_mach or r_ref > r_mach:
        warnings.append(
            "จุดอ้างอิงขยับมากเทียบกับจุดบนเครื่อง "
            "(p2p %.4f เทียบ %.4f px, rms %.4f เทียบ %.4f px) "
            "แปลว่าจุดอ้างอิงอาจไม่ได้ยึดกับโลกจริง "
            "หรือการสั่นของเครื่องเล็กกว่าพื้นสัญญาณรบกวน "
            "ผลการหักล้างการสั่นของกล้องจะเชื่อถือไม่ได้"
            % (p_ref, p_mach, r_ref, r_mach))
    return warnings
