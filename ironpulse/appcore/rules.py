import numpy as np

UNBALANCE = "ความไม่สมดุลของมวลหมุน"
LOOSENESS = "การหลวมของจุดยึด"
RUBBING = "การเสียดสีหรือตลับลูกปืนสึกหรอ"
BLADE = "การอุดตันทางไหลหรือใบพัดเสียหาย"
SPEED = "ตรวจสอบความเร็วรอบก่อน อาจไม่ใช่ความผิดปกติ"
UNKNOWN = "ผิดปกติแต่ระบุชนิดไม่ได้"

ORDER_FEATURES = ["r05", "r1", "r2", "r3"]

EPS = 1e-12


class RuleError(Exception):
    pass


def std_floor(mean, std, floor_frac):
    return max(float(std), abs(float(mean)) * float(floor_frac), EPS)


def zscore(value, mean, std, floor_frac):
    return (float(value) - float(mean)) / std_floor(mean, std, floor_frac)


def zmap(vector, names, baseline_mean, baseline_std, app_cfg, signal):
    floor_frac = float(app_cfg["rules"]["z_std_floor_frac"])
    vector = np.asarray(vector, dtype=np.float64)
    baseline_mean = np.asarray(baseline_mean, dtype=np.float64)
    baseline_std = np.asarray(baseline_std, dtype=np.float64)
    if not (len(vector) == len(names) == len(baseline_mean) == len(baseline_std)):
        raise RuleError(
            "ความยาวไม่ตรงกัน vector %d ชื่อ %d baseline mean %d sd %d "
            "มักเกิดจาก baseline สร้างด้วยจำนวนมาร์กเกอร์ไม่เท่ากับคลิปที่ตรวจ"
            % (len(vector), len(names), len(baseline_mean), len(baseline_std)))
    prefix = signal + "."
    out = {}
    for i, nm in enumerate(names):
        if not nm.startswith(prefix):
            continue
        out[nm[len(prefix):]] = zscore(vector[i], baseline_mean[i],
                                       baseline_std[i], floor_frac)
    if not out:
        raise RuleError("ไม่มี feature ของสัญญาณ %s ใน feature vector" % signal)
    return out


def fmt(z, key):
    return "z(%s) = %+.2f" % (key, z[key])


def evaluate(z, app_cfg, available=None):
    k = float(app_cfg["rules"]["z_threshold"])
    frac = float(app_cfg["rules"]["z_broadband_frac"])
    available = available or {}
    hits = []
    skipped = []

    def has(*keys):
        for key in keys:
            if key not in z:
                return False
            if available.get(key) is False:
                return False
        return True

    if has(*ORDER_FEATURES):
        ranked = sorted(((z[key], key) for key in ORDER_FEATURES), reverse=True)
        if ranked[0][1] == "r1" and z["r1"] >= k:
            hits.append((UNBALANCE,
                         [fmt(z, "r1"),
                          "สูงสุดในกลุ่ม order",
                          "รองลงมา %s" % fmt(z, ranked[1][1])]))
    else:
        skipped.append(UNBALANCE)

    if has("r2", "r3", "r05"):
        if (z["r2"] >= k or z["r3"] >= k) and z["r05"] >= k:
            hits.append((LOOSENESS, [fmt(z, "r2"), fmt(z, "r3"), fmt(z, "r05")]))
    else:
        skipped.append(LOOSENESS)

    if has("E_hi", "kurt"):
        if z["E_hi"] >= k and z["kurt"] >= k:
            hits.append((RUBBING, [fmt(z, "E_hi"), fmt(z, "kurt")]))
    else:
        skipped.append(RUBBING)

    if has("rbp"):
        if abs(z["rbp"]) >= k:
            hits.append((BLADE, [fmt(z, "rbp")]))
    else:
        skipped.append(BLADE)

    fired = [key for key in z if abs(z[key]) >= k]
    if len(z) and len(fired) >= frac * len(z):
        signs = set(np.sign(z[key]) for key in fired)
        if len(signs) == 1:
            hits.append((SPEED, ["%d/%d feature เกิน %.1f sd ไปทางเดียวกัน"
                                 % (len(fired), len(z), k)]))
    return dict(threshold=k, hits=hits, skipped=skipped,
                n_fired=len(fired), n_features=len(z), z=z)


def verdict(result, anomalous):
    if result["hits"]:
        return result["hits"][0][0]
    return UNKNOWN if anomalous else ""


def report_lines(result):
    lines = ["เกณฑ์ z >= %.1f sd ของ baseline" % result["threshold"]]
    for name, ev in result["hits"]:
        lines.append("%s : %s" % (name, " · ".join(ev)))
    if not result["hits"]:
        lines.append("ไม่เข้ากฎใดเลย (%d/%d feature เกินเกณฑ์)"
                     % (result["n_fired"], result["n_features"]))
    for name in result["skipped"]:
        lines.append("ข้ามกฎ %s เพราะ feature ที่ต้องใช้ยังไม่ได้วัด" % name)
    return lines
