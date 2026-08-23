import os, re, datetime
import numpy as np
from . import config

ROOT = os.path.join(config.ROOT, "datasets", "raw", "ims", "extracted")
FS = 20000.0
RPM = 2000.0
F0 = RPM / 60.0
TESTS = {
    "2nd_test": dict(n_channels=4, failed_bearing=1, fault="outer_race",
                     channels={1: 0, 2: 1, 3: 2, 4: 3}),
    "3rd_test": dict(n_channels=4, failed_bearing=3, fault="outer_race",
                     channels={1: 0, 2: 1, 3: 2, 4: 3}),
    "1st_test": dict(n_channels=8, failed_bearing=3, fault="inner_race",
                     channels={1: 0, 2: 2, 3: 4, 4: 6}),
}
STAMP = re.compile(r"^(\d{4})\.(\d{2})\.(\d{2})\.(\d{2})\.(\d{2})\.(\d{2})$")


ALIASES = {"3rd_test": ["3rd_test", "4th_test"]}


def test_dir(name):
    for cand in ALIASES.get(name, [name]):
        p = os.path.join(ROOT, cand)
        if not os.path.isdir(p):
            continue
        for inner in [os.path.join(p, cand), os.path.join(p, "txt"), p]:
            if not os.path.isdir(inner):
                continue
            if any(STAMP.match(f) for f in os.listdir(inner)[:50]):
                return inner
    return None


def parse_stamp(fn):
    m = STAMP.match(fn)
    if not m:
        return None
    y, mo, d, h, mi, s = [int(x) for x in m.groups()]
    return datetime.datetime(y, mo, d, h, mi, s)


def index(name="2nd_test"):
    d = test_dir(name)
    if not d:
        return []
    out = []
    for fn in sorted(os.listdir(d)):
        t = parse_stamp(fn)
        if t is None:
            continue
        out.append(dict(test=name, path=os.path.join(d, fn), stamp=t, name=fn))
    out.sort(key=lambda r: r["stamp"])
    if out:
        t0 = out[0]["stamp"]
        tend = out[-1]["stamp"]
        for r in out:
            r["hours_from_start"] = (r["stamp"] - t0).total_seconds() / 3600.0
            r["hours_to_end"] = (tend - r["stamp"]).total_seconds() / 3600.0
    return out


def load(rec, bearing=None):
    meta = TESTS[rec["test"]]
    b = bearing or meta["failed_bearing"]
    col = meta["channels"][b]
    a = np.loadtxt(rec["path"], dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return a[:, col], FS, F0
