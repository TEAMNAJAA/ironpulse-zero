import os
import sys
import re
import csv
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
REPO = os.path.dirname(APP)
sys.path.insert(0, REPO)
sys.path.insert(0, APP)
os.environ.setdefault("IRONPULSE_PROJECT_ROOT", APP)

import imageio_ffmpeg
from appcore import video

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
ROOT = r"C:\Users\more_\Downloads\Arise H4"
GROUPS = [("normal", "normal", 0),
          (u"\u0e19\u0e49\u0e2d\u0e22\u0e01\u0e27\u0e48\u0e32" u"3"
           u"\u0e01\u0e23\u0e31\u0e21", "under3g", 1),
          (u"3\u0e01\u0e23\u0e31\u0e21", "3g", 2),
          ("calibration_ruler", "calibration", -1)]
OUT = os.path.join(HERE, "inventory.csv")
CREATION = re.compile(r"creation_time\s*:\s*(\S+)")


def creation_time(path):
    r = subprocess.run([FFMPEG, "-hide_banner", "-i", path],
                       capture_output=True, text=True, errors="replace")
    m = CREATION.search(r.stderr or "")
    return m.group(1) if m else ""


def main():
    rows = []
    for folder, label, severity in GROUPS:
        d = os.path.join(ROOT, folder)
        if not os.path.isdir(d):
            print("missing folder", d)
            continue
        for name in sorted(os.listdir(d)):
            if not name.lower().endswith(".mov"):
                continue
            path = os.path.join(d, name)
            info = video.probe(path)
            rows.append(dict(clip=os.path.splitext(name)[0], group=label,
                             severity=severity, folder=folder,
                             created=creation_time(path),
                             codec=info["codec"], width=info["width"],
                             height=info["height"], n_frames=info["n_frames"],
                             meta_fps=info["meta_fps"],
                             container_sec=info["container_sec"],
                             has_audio=info["has_audio"],
                             bytes=os.path.getsize(path), path=path))
            print("%-12s %-12s %5d frames  %s" %
                  (rows[-1]["clip"], label, info["n_frames"], rows[-1]["created"]),
                  flush=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print()
    print("wrote", OUT, len(rows), "clips")


if __name__ == "__main__":
    main()
