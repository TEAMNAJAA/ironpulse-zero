import os
import sys
import io
import argparse
import yaml
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
REPO = os.path.dirname(APP)
sys.path.insert(0, REPO)
sys.path.insert(0, APP)
os.environ.setdefault("IRONPULSE_PROJECT_ROOT", APP)

from appcore import video, calibrate


def parse_pair(s):
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            "รูปแบบต้องเป็น x1,y1,x2,y2,mm เช่น 300,900,300,1200,50")
    x1, y1, x2, y2, mm = [float(p) for p in parts]
    return ((x1, y1), (x2, y2), mm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--setup-id", required=True)
    ap.add_argument("--pair", action="append", type=parse_pair, default=[])
    ap.add_argument("--notes", default="")
    ap.add_argument("--dump-frame", default="")
    args = ap.parse_args()

    cfg = yaml.safe_load(io.open(os.path.join(APP, "config.yaml"), encoding="utf-8"))
    cal_path = os.path.join(APP, cfg["scale"]["calibration_file"])
    flow_px = float(cfg["scale"]["flow_resolution_px"])

    if args.dump_frame:
        g = video.first_frame(args.clip)
        cv2.imwrite(args.dump_frame, g)
        print("เขียนเฟรมแรกไปที่ %s ขนาด %dx%d" % (args.dump_frame, g.shape[1], g.shape[0]))
        print("เปิดไฟล์นี้แล้วอ่านพิกัดสองจุดบนไม้บรรทัดที่รู้ระยะจริง")
        if not args.pair:
            return 0

    if not args.pair:
        print("ต้องระบุ --pair อย่างน้อยหนึ่งชุด รูปแบบ x1,y1,x2,y2,mm")
        return 2

    cal = calibrate.from_clip(args.clip, args.pair, args.setup_id, args.notes)
    calibrate.store(cal, cal_path)

    print("setup_id        : %s" % cal["setup_id"])
    print("clip            : %s" % cal["source"])
    print("resolution      : %dx%d" % (cal["width"], cal["height"]))
    print("จำนวนคู่จุด      : %d" % cal["n_pairs"])
    print("px per mm       : %.4f" % cal["px_per_mm"])
    print("um per px       : %.2f" % cal["um_per_px"])
    if cal["n_pairs"] > 1:
        print("สเปรดระหว่างคู่  : %.2f %% (%.4f ถึง %.4f px/mm)"
              % (100 * cal["spread_frac"], cal["px_per_mm_min"], cal["px_per_mm_max"]))
    floor = calibrate.detection_floor_um(cal, flow_px)
    print("ขีดจำกัดการวัด   : ความละเอียด flow %.3f px = %.2f um" % (flow_px, floor))
    print("เขียนไปที่       : %s" % cal_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
