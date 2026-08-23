import os
import sys
import io
import csv
import yaml
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
REPO = os.path.dirname(APP)
sys.path.insert(0, REPO)
sys.path.insert(0, APP)
os.environ.setdefault("IRONPULSE_PROJECT_ROOT", APP)

from appcore import rules

NORMAL = ["normal_01", "normal_02", "normal_03"]
UNBAL = ["unbal_01", "unbal_02", "unbal_03"]
SIGNALS = ["d_machine_x", "d_machine_y"]
NPZ = os.path.join(HERE, "pilot2_data.npz")
CSV = os.path.join(HERE, "pilot2_raw.csv")
LINE = "=" * 96


def main():
    cfg = yaml.safe_load(io.open(os.path.join(APP, "config.yaml"), encoding="utf-8"))
    d = np.load(NPZ, allow_pickle=True)
    clips = list(d["clips"])
    names = list(d["names"])
    V = d["vectors"]
    gi = {c: i for i, c in enumerate(clips)}
    ni = [gi[c] for c in NORMAL]
    mean = V[ni].mean(axis=0)
    std = V[ni].std(axis=0, ddof=1)
    rows = {r["clip"]: r for r in csv.DictReader(io.open(CSV, encoding="utf-8"))}
    k = float(cfg["rules"]["z_threshold"])

    print(LINE)
    print("z-score rule check on the pilot2 clips")
    print(LINE)
    print("baseline = %d normal clips (config min_baseline_clips = %s)"
          % (len(NORMAL), cfg["model"]["min_baseline_clips"]))
    print("z threshold k = %.1f sd, std floor = %.0f%% of the mean"
          % (k, 100 * float(cfg["rules"]["z_std_floor_frac"])))
    print("this is a smoke test of the rule mechanics, not a validated threshold")
    print()

    for signal in SIGNALS:
        print(LINE)
        print("signal %s" % signal)
        print(LINE)
        for c in clips:
            avail = {"rbp": rows[c]["rbp_available"] == "True"}
            z = rules.zmap(V[gi[c]], names, mean, std, cfg, signal)
            res = rules.evaluate(z, cfg, avail)
            top = sorted(z.items(), key=lambda kv: -abs(kv[1]))[:4]
            print("%-11s %-8s  %s" %
                  (c, "normal" if c in NORMAL else "unbal",
                   "  ".join("%s=%+.1f" % (a, b) for a, b in top)))
            for line in rules.report_lines(res):
                print("            %s" % line)
            print()


if __name__ == "__main__":
    main()
