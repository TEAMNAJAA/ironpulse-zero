import os, sys, csv
import numpy as np

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)
sys.path.insert(0, os.path.dirname(_PROJ))
from core import config, signals as S, detectors, evaluate as EV, features as F

CFG = config.CFG
np.random.seed(CFG["seed"])


def cwru_index_48k():
    idx = S.cwru_index()
    keep = [r for r in idx if r["page"] in ("normal-baseline-data",
                                            "48k-drive-end-bearing-fault-data")]
    return sorted(keep, key=lambda r: r["file_id"])


def cwru_index_12k_de():
    idx = S.cwru_index()
    keep = [r for r in idx if r["page"] in ("normal-baseline-data",
                                            "12k-drive-end-bearing-fault-data")]
    return sorted(keep, key=lambda r: r["file_id"])


def maf_index_all():
    return sorted(S.maf_index(), key=lambda r: r["file_id"])


def write_csv(path, rows, fieldnames=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        open(path, "w", encoding="utf-8").write("")
        return
    fn = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fn, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("wrote", path, len(rows), "rows")


def table(key, name):
    return os.path.join(CFG["paths"]["tables"], name)


def figure(name):
    os.makedirs(CFG["paths"]["figures"], exist_ok=True)
    return os.path.join(CFG["paths"]["figures"], name)
