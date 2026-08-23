import os
import re
import subprocess
import numpy as np
import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
_OUT_DIMS = re.compile(r"Stream #\d+:\d+.*?: Video: rawvideo.*?, (\d+)x(\d+)")
_IN_VIDEO = re.compile(r"Stream #\d+:\d+.*?: Video: (\w+).*?, (\d+)x(\d+)")
_IN_FPS = re.compile(r"(\d+(?:\.\d+)?) fps")
_IN_AUDIO = re.compile(r"Stream #\d+:\d+.*?: Audio: ")
_DURATION = re.compile(r"Duration: (\d+):(\d+):(\d+\.\d+)")
_FRAME = re.compile(r"frame=\s*(\d+)")
_TIME = re.compile(r"time=(\d+):(\d+):(\d+\.\d+)")


class VideoError(Exception):
    pass


def _run(args):
    return subprocess.run([FFMPEG] + args, capture_output=True, text=True,
                          errors="replace")


def _banner(path):
    r = _run(["-hide_banner", "-i", path])
    return r.stderr or ""


def probe(path):
    if not os.path.isfile(path):
        raise VideoError("ไม่พบไฟล์วิดีโอ: %s" % path)
    banner = _banner(path)
    mv = _IN_VIDEO.search(banner)
    if not mv:
        raise VideoError(
            "เปิดไฟล์วิดีโอไม่ได้ อาจไม่ใช่ไฟล์วิดีโอหรือไฟล์เสียหาย: %s" % os.path.basename(path))
    codec = mv.group(1)
    line = banner[mv.start():banner.find("\n", mv.start())]
    mf = _IN_FPS.search(line)
    meta_fps = float(mf.group(1)) if mf else None
    md = _DURATION.search(banner)
    container_sec = (int(md.group(1)) * 3600 + int(md.group(2)) * 60 +
                     float(md.group(3))) if md else None
    has_audio = bool(_IN_AUDIO.search(banner))
    probe_out = _run(["-v", "info", "-i", path, "-frames:v", "1",
                      "-vf", "format=gray", "-f", "rawvideo", "-pix_fmt", "gray", "-"])
    mo = _OUT_DIMS.search(probe_out.stderr or "")
    if not mo:
        raise VideoError("ถอดรหัสวิดีโอไม่สำเร็จ: %s" % os.path.basename(path))
    width, height = int(mo.group(1)), int(mo.group(2))
    count = _run(["-v", "error", "-stats", "-i", path, "-map", "0:v:0",
                  "-c", "copy", "-f", "null", "-"])
    frames = None
    for m in _FRAME.finditer(count.stderr or ""):
        frames = int(m.group(1))
    if frames is None:
        raise VideoError("นับจำนวนเฟรมไม่ได้: %s" % os.path.basename(path))
    return dict(path=path, codec=codec, width=width, height=height,
                n_frames=frames, meta_fps=meta_fps, has_audio=has_audio,
                container_sec=container_sec)


def audio_seconds(path):
    r = _run(["-v", "error", "-stats", "-i", path, "-map", "0:a:0",
              "-f", "null", "-"])
    last = None
    for m in _TIME.finditer(r.stderr or ""):
        last = m
    if last is None:
        return None
    return int(last.group(1)) * 3600 + int(last.group(2)) * 60 + float(last.group(3))


def audio_stretch(path, true_fps, n_frames):
    a = audio_seconds(path)
    if not a or not n_frames:
        return None
    return a / (n_frames / float(true_fps))


def frames_used(info, cfg):
    cap = cfg["capture"].get("max_frames")
    if not cap:
        return info["n_frames"]
    return min(info["n_frames"], int(cap))


def check_usable(info, cfg):
    warnings = []
    min_frames = int(cfg["capture"]["min_frames"])
    if info["n_frames"] < min_frames:
        raise VideoError(
            "คลิปสั้นเกินไป มี %d เฟรม ต้องการอย่างน้อย %d เฟรม "
            "ที่ %.0f fps คือประมาณ %.1f วินาที "
            "iOS ตัดหัวท้ายคลิปละประมาณ %.1f วินาทีตอน export "
            "จึงควรอัดอย่างน้อย %.0f วินาที"
            % (info["n_frames"], min_frames, cfg["capture"]["default_fps"],
               min_frames / float(cfg["capture"]["default_fps"]),
               cfg["capture"]["ios_export_trim_sec"],
               cfg["capture"]["min_record_sec"]))
    used = frames_used(info, cfg)
    if used < info["n_frames"]:
        warnings.append(
            "คลิปมี %d เฟรม ระบบใช้ %d เฟรมแรกเพื่อให้ทุกคลิปมีความละเอียดเชิงความถี่เท่ากันที่ %.4f Hz"
            % (info["n_frames"], used, float(cfg["capture"]["default_fps"]) / used))
    true_fps = float(cfg["capture"]["default_fps"])
    mf = info.get("meta_fps")
    if mf:
        ratio = abs(mf - true_fps) / true_fps
        if ratio > float(cfg["capture"]["fps_mismatch_warn_ratio"]):
            warnings.append(
                "metadata ของไฟล์บอก %.1f fps แต่ระบบใช้ค่าที่วัดจริง %.1f fps "
                "ในการคำนวณทั้งหมด (ไฟล์สโลว์โมชั่นจาก iPhone บอก fps ผิดเป็นปกติ)"
                % (mf, true_fps))
    return warnings


def crop_filter(crop):
    if not crop:
        return "format=gray"
    x, y, w, h = crop
    return "crop=%d:%d:%d:%d,format=gray" % (w, h, x, y)


def frames(path, crop=None, width=None, height=None, limit=None):
    if width is None or height is None:
        info = probe(path)
        width, height = info["width"], info["height"]
    if crop:
        w, h = crop[2], crop[3]
    else:
        w, h = width, height
    args = [FFMPEG, "-v", "error", "-i", path, "-vf", crop_filter(crop)]
    if limit:
        args += ["-frames:v", str(int(limit))]
    args += ["-f", "rawvideo", "-pix_fmt", "gray", "-"]
    nbytes = w * h
    proc = subprocess.Popen(args, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, bufsize=nbytes * 4)
    try:
        while True:
            buf = proc.stdout.read(nbytes)
            if not buf or len(buf) < nbytes:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w)
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.wait()


def first_frame(path, crop=None):
    for f in frames(path, crop=crop):
        return f.copy()
    raise VideoError("อ่านเฟรมแรกไม่ได้: %s" % os.path.basename(path))
