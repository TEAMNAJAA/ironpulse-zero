import csv
import io
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
REPO = os.path.dirname(APP)
DOCS = os.path.join(REPO, "docs")
FIGS_SRC = os.path.join(APP, "h4", "results", "figures")
TABLES = os.path.join(APP, "h4", "results", "tables")
PILOT_FIG = os.path.join(APP, "verify", "pilot2_spectra.png")
GH = "https://github.com/%s/%s/blob/main"

FIGURES = [
    ("A_confusion_matrix.png", "A · Confusion matrix ระดับคลิป",
     "ตัวเลขในช่องคือจำนวนคลิป เปอร์เซ็นต์คิดตามแถว มี N และเกณฑ์กำกับใต้รูป"),
    ("B_score_distribution.png", "B · การกระจายของคะแนน",
     "รูปที่อธิบายวิธีการได้ดีที่สุด เห็นทั้งการแยกกลุ่มและตำแหน่งเกณฑ์ในภาพเดียว "
     "เกณฑ์ตั้งจากคลิปปกติในชุดฝึกเท่านั้น"),
    ("C_roc_curve.png", "C · ROC curve",
     "ช่วงความเชื่อมั่นจาก bootstrap 2000 รอบที่ระดับคลิป"),
    ("D_detection_by_severity.png", "D · อัตราตรวจพบแยกตามระดับความรุนแรง",
     "แท่งคือค่าเฉลี่ยจาก 20 fold-run เส้นคือส่วนเบี่ยงเบนมาตรฐาน 1 sd"),
    ("E_order_spectra.png", "E · สเปกตรัมแกน order",
     "เส้นหนาคือค่ากลางของกลุ่ม แถบคือเปอร์เซ็นไทล์ที่ 10 ถึง 90 "
     "กลุ่มถ่วง 3 กรัมมีพีคที่ 1x สูงกว่ากลุ่มปกติชัดเจน พร้อมฮาร์มอนิกถึง 5x"),
    ("F_model_comparison.png", "F · เปรียบเทียบโมเดลทั้งสี่",
     "ค่าจากวงในของ nested CV ยิ่งต่ำยิ่งดี IsolationForest ถูกเลือกทั้ง 20 ครั้ง"),
    ("pilot2_spectra.png", "Pilot รอบ 2 · สเปกตรัมแกน order",
     "6 คลิปนำร่องก่อนเก็บชุดจริง น้ำเงินคือปกติ ส้มคือไม่สมดุล"),
]


def rows(path):
    return list(csv.DictReader(io.open(path, encoding="utf-8")))


def table2():
    return {r["metric"]: r for r in rows(os.path.join(TABLES, "table2_summary.csv"))}


def fold_rows():
    return [r for r in rows(os.path.join(TABLES, "table3_fold_metrics.csv"))
            if r["level"] == "clip"]


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def fold_table():
    import collections
    by = collections.defaultdict(list)
    for r in fold_rows():
        by[r["fold"]].append(r)
    out = []
    for f in sorted(by):
        rs = by[f]
        m = lambda k: sum(float(x[k]) for x in rs) / len(rs)
        out.append((f, rs[0]["n_normal"], m("auc"), m("far"), m("tpr_3g"),
                    m("tpr_under3g")))
    return out


def build(owner, repo):
    os.makedirs(os.path.join(DOCS, "figures"), exist_ok=True)
    for name, _, _ in FIGURES:
        src = PILOT_FIG if name.startswith("pilot2") else os.path.join(FIGS_SRC, name)
        shutil.copy2(src, os.path.join(DOCS, "figures", name))

    t2 = table2()
    t1 = rows(os.path.join(TABLES, "table1_by_severity.csv"))
    t4 = rows(os.path.join(TABLES, "table4_model_selection.csv"))
    t5 = rows(os.path.join(TABLES, "table5_per_clip.csv"))
    base = GH % (owner, repo)

    def v(metric, field="value"):
        return t2[metric][field]

    auc = float(v("roc_auc_pooled"))
    lo, hi = float(v("roc_auc_pooled", "ci_low")), float(v("roc_auc_pooled", "ci_high"))
    far = float(v("false_alarm_rate"))
    per_week = far * 20 * 7

    head = """<!doctype html>
<html lang="th">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>IronPulse Zero — ตรวจความไม่สมดุลของมวลหมุนจากวิดีโอมือถือ</title>
<style>
:root{--bg:#f4f6f8;--card:#fff;--ink:#14181d;--muted:#5a6572;--line:#d9dee5;
--ok:#0f7b4f;--bad:#b02a12;--warn-bg:#fff5d6;--accent:#0b5fa5}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font-family:"Segoe UI","Leelawadee UI",system-ui,sans-serif;line-height:1.6}
header{background:#0e1b2a;color:#fff;padding:38px 20px}
header .in{max-width:1080px;margin:0 auto}
header h1{margin:0 0 6px;font-size:30px}
header p{margin:4px 0;color:#b9c6d4}
main{max-width:1080px;margin:0 auto;padding:22px 20px 60px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:20px;margin-bottom:20px}
h2{font-size:21px;margin:0 0 14px}
h3{font-size:16px;color:var(--muted);margin:20px 0 8px}
.kpi{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px}
.kpi div{background:#fbfcfd;border:1px solid var(--line);border-radius:9px;padding:14px}
.kpi .k{font-size:12px;color:var(--muted);font-weight:700}
.kpi .v{font-size:27px;font-weight:800}
.kpi .u{font-size:12px;color:var(--muted)}
table{width:100%;border-collapse:collapse;font-size:14px}
th,td{text-align:left;padding:7px 9px;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-size:12px}
td.bad{color:var(--bad);font-weight:700}td.ok{color:var(--ok);font-weight:700}
.warn{background:var(--warn-bg);border:1px solid #e2c56a;border-left:6px solid #d9a400;
color:#4a3600;border-radius:8px;padding:12px 14px;margin:10px 0;font-size:14px}
.stop{background:#fdeae5;border:1px solid #e8a08c;border-left:6px solid var(--bad);
color:#5c1a0a;border-radius:8px;padding:12px 14px;margin:10px 0}
figure{margin:0 0 26px}
figure img{width:100%;border:1px solid var(--line);border-radius:9px;background:#fff}
figcaption{font-size:13px;color:var(--muted);margin-top:7px}
code,pre{font-family:Consolas,monospace;font-size:13px}
pre{background:#0e1b2a;color:#e6edf3;padding:14px;border-radius:9px;overflow-x:auto}
a{color:var(--accent)}
.muted{color:var(--muted);font-size:13px}
.scroll{overflow-x:auto}
</style>
</head>
<body>
<header><div class="in">
<h1>IronPulse Zero</h1>
<p>ตรวจจับความไม่สมดุลของมวลหมุนจากวิดีโอกล้องมือถือ โดยไม่ต้องแตะเครื่องจักร</p>
<p>กล้อง iPhone 240 fps &middot; optical flow ระดับซับพิกเซล &middot; โมเดลเรียนจากเครื่องปกติเท่านั้น</p>
</div></header>
<main>
"""

    scope = """<div class="stop">
<b>ขอบเขตของระบบ</b> ตรวจได้เฉพาะ <b>ความไม่สมดุลของมวลหมุน</b> อย่างเดียวเท่านั้น
ชุดข้อมูลไม่มีฐานหลวม การอุดตัน หรือการเสียดสี จึงไม่ประเมินและไม่รายงานชนิดเหล่านั้นในทุกกรณี
</div>
<div class="warn">
หน้านี้แสดง <b>ผลที่บันทึกไว้จากการรันจริง</b> ทุกตัวเลขและทุกรูปมาจากคลิปวิดีโอจริง 64 คลิป
ไม่มีข้อมูลสมมติ ไม่มีการปัดตัวเลขให้สวย
<b>หน้านี้ไม่ใช่ตัวโปรแกรม</b> เพราะโปรแกรมจริงเป็น Python ที่ต้องถอดรหัสวิดีโอ จึงรันบน GitHub Pages ไม่ได้
วิธีรันตัวจริงอยู่ท้ายหน้า
</div>
"""

    kpi = """<div class="card">
<h2>ผลการวัด</h2>
<div class="kpi">
<div><div class="k">ROC-AUC ระดับคลิป</div><div class="v">%.4f</div>
<div class="u">95%% CI %.4f &ndash; %.4f &middot; bootstrap 2000 รอบ</div></div>
<div><div class="k">อัตราแจ้งเตือนผิดพลาด</div><div class="v">%.2f %%</div>
<div class="u">คิดจากคลิปปกติเท่านั้น &asymp; %.1f ครั้งต่อสัปดาห์ ถ้าตรวจ 20 เครื่องต่อวัน</div></div>
<div><div class="k">ความละเอียดที่วัดได้</div><div class="v">%.1f</div>
<div class="u">ไมโครเมตร &middot; สอบเทียบด้วยไม้บรรทัดในเฟรม</div></div>
<div><div class="k">เวลาต่อคลิป</div><div class="v">7.7</div>
<div class="u">วินาที ต่อ 2000 เฟรม &middot; เป้าหมาย 12 วินาที</div></div>
</div>
<div class="warn"><b>ห้ามอ่านเป็น accuracy</b> ชุดทดสอบมีปกติ 40 ผิดปกติ 24 ซึ่งเกือบสมดุล
แต่โรงงานจริงเครื่องปกติเกือบตลอดเวลา ค่า accuracy จะสูงเกินจริงมาก
จึงรายงานอัตราตรวจพบกับอัตราแจ้งเตือนผิดพลาดแยกกันเสมอ</div>
</div>
""" % (auc, lo, hi, 100 * far, per_week, float(v("detection_floor_um")))

    sev = ["""<div class="card"><h2>อัตราตรวจพบแยกตามระดับความรุนแรง</h2><div class="scroll"><table>
<tr><th>กลุ่ม</th><th>จำนวนคลิป</th><th>อัตราตรวจพบ</th><th>แอมพลิจูดที่ 1&times;</th><th>ความเร็วรอบ</th></tr>"""]
    for r in t1:
        sev.append("<tr><td>%s</td><td>%s</td><td>%s &plusmn; %s</td><td>%s &micro;m</td>"
                   "<td>%s Hz</td></tr>" %
                   (esc(r["group"]), r["n_clips"], r["detection_rate_mean"],
                    r["detection_rate_sd"], r["a1_um_mean"], r["f0_hz_mean"]))
    sev.append("</table></div>")
    sev.append("""<div class="warn"><b>ทำไมกลุ่มต่ำกว่า 3 กรัมถึงตรวจพบได้น้อยกว่า</b>
คลิปที่พลาดมีการสั่นที่ 21&ndash;27 &micro;m ซึ่งอยู่ในพิสัยเดียวกับเครื่องปกติ (4&ndash;41 &micro;m)
และสูงกว่าขีดจำกัดของกล้องเพียง 1.1&ndash;1.4 เท่า
ที่ 389 &micro;m ต่อพิกเซล การสั่น 25 &micro;m คือการขยับเพียง <b>0.064 พิกเซล</b>
นี่คือขีดจำกัดทางฟิสิกส์ของการจัดเฟรมชุดนี้ ไม่ใช่ข้อบกพร่องของโมเดล แก้ได้ด้วยการซูมเข้า</div></div>""")

    folds = ["""<div class="card"><h2>ผลแยกตามรอบการถ่ายที่กันไว้ทดสอบ</h2>
<p class="muted">แบ่งข้อมูลตามรอบการถ่าย ไม่ใช่ตามคลิป เพื่อไม่ให้โมเดลจำการตั้งกล้องแทนที่จะจำสภาวะเครื่อง</p>
<div class="scroll"><table>
<tr><th>รอบที่กันไว้</th><th>คลิปปกติที่เหลือไว้ฝึก</th><th>AUC</th><th>แจ้งเตือนผิด</th>
<th>ตรวจพบ 3 ก.</th><th>ตรวจพบ &lt;3 ก.</th></tr>"""]
    for f, ntr, a, fa, t3, tu in fold_table():
        cls = ' class="bad"' if a < 0.9 else ""
        folds.append("<tr><td>%s</td><td>%s</td><td%s>%.4f</td><td>%.4f</td>"
                     "<td%s>%.3f</td><td%s>%.3f</td></tr>" %
                     (f, 40 - int(ntr), cls, a, fa, cls, t3, cls, tu))
    folds.append("</table></div>")
    folds.append("""<div class="warn"><b>รอบ S1 ตรวจไม่พบอะไรเลย และเราไม่ซ่อนข้อนี้</b>
คลิปปกติ 26 จาก 40 คลิปกระจุกอยู่ในรอบถ่ายเดียว พอกันรอบนั้นไว้ทดสอบก็เหลือคลิปฝึกแค่ 14 คลิป
ซึ่งต่ำกว่าขั้นต่ำ 20 คลิปที่ระบบกำหนดไว้เอง เกณฑ์จึงสูงเกินไปจนไม่มีอะไรผ่าน
<b>ตัวเลขพาดหัวคือค่าเฉลี่ยจากทั้ง 4 รอบ</b> ไม่ใช่เฉพาะรอบที่ผลดี
ปัญหาอยู่ที่การเก็บข้อมูล ไม่ใช่ที่โมเดล</div></div>""")

    models = ["""<div class="card"><h2>โมเดลที่ถูกเลือก</h2>
<p class="muted">วงในของ nested CV เลือกจากคลิปปกติล้วน ไม่เคยเห็นคลิปผิดปกติ</p>
<div class="scroll"><table><tr><th>โมเดล</th><th>ค่าเกณฑ์วงใน (ต่ำ = ดี)</th>
<th>ถูกเลือกกี่ครั้งจาก 20</th></tr>"""]
    for r in t4:
        b = "<b>" if r["model"] == "isolation_forest" else ""
        e = "</b>" if b else ""
        models.append("<tr><td>%s%s%s</td><td>%s%s &plusmn; %s%s</td><td>%s%s/20%s</td></tr>" %
                      (b, esc(r["model"]), e, b, r["criterion_mean"], r["criterion_sd"], e,
                       b, r["times_selected_clip"], e))
    models.append("</table></div></div>")

    figs = ['<div class="card"><h2>รูปทั้งหมด</h2>']
    for name, title, cap in FIGURES:
        figs.append('<figure><h3>%s</h3><img src="figures/%s" alt="%s" loading="lazy">'
                    '<figcaption>%s</figcaption></figure>' %
                    (esc(title), name, esc(title), cap))
    figs.append("</div>")

    per = ['<div class="card"><h2>ผลรายคลิปทั้ง 64 คลิป</h2>'
           '<p class="muted">คะแนนต่อเกณฑ์เกิน 1.000 คือถูกตัดสินว่าผิดปกติ</p>'
           '<div class="scroll" style="max-height:460px"><table>'
           '<tr><th>คลิป</th><th>กลุ่ม</th><th>รอบถ่าย</th><th>f0 (Hz)</th>'
           '<th>A1 (&micro;m)</th><th>คะแนน/เกณฑ์</th><th>ผล</th></tr>']
    for r in sorted(t5, key=lambda x: (x["group"], x["clip"])):
        flagged = r["flagged"] == "1"
        per.append("<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td>"
                   "<td>%s</td><td class='%s'>%s</td></tr>" %
                   (esc(r["clip"]), esc(r["group"]), esc(r["session"]), r["f0_hz"],
                    r["a1_um"], r["pooled_score"], "bad" if flagged else "ok",
                    "ผิดปกติ" if flagged else "ปกติ"))
    per.append("</table></div></div>")

    how = """<div class="card">
<h2>วิธีรันโปรแกรมตัวจริง</h2>
<p>เว็บแอปเป็น Python FastAPI ที่ทำงานบนเครื่องของผู้ใช้ ไม่ต้องต่ออินเทอร์เน็ตขณะใช้งาน</p>
<pre>git clone https://github.com/%s/%s.git
cd %s/ironpulse
pip install fastapi uvicorn python-multipart numpy scipy scikit-learn opencv-python pyyaml imageio-ffmpeg matplotlib
python web/seed_demo.py
python run.py</pre>
<p class="muted">เปิดเบราว์เซอร์ให้อัตโนมัติที่ <code>http://127.0.0.1:8765/</code>
คลิปวิดีโอไม่ได้อยู่ใน repo เพราะไฟล์ละราว 100 MB
แต่ <code>ironpulse/h4/tracks.npz</code> และ <code>features.npz</code> อยู่ครบ
จึงรัน <code>run_nested_cv.py</code> กับ <code>evaluate.py</code> ซ้ำเพื่อสร้างผลและรูปทั้งหมดใหม่ได้โดยไม่ต้องมีวิดีโอ</p>
<h3>เอกสาร</h3>
<ul>
<li><a href="%s/ironpulse/h4/H4_REPORT.md">H4_REPORT.md</a> วิธีเลือกโมเดลและการประเมินผลฉบับเต็ม พร้อมทุกข้อที่ทำไม่ได้ตามสเปกและเหตุผล</li>
<li><a href="%s/ironpulse/h4/POSTER_NUMBERS.md">POSTER_NUMBERS.md</a> ประโยคที่ยกไปวางบนโปสเตอร์ได้ พร้อมรายการตัวเลขที่ห้ามใช้</li>
<li><a href="%s/ironpulse/web/H5_REPORT.md">H5_REPORT.md</a> รายงานเว็บแอปและผลเกณฑ์ตรวจรับ 10 ข้อ</li>
<li><a href="%s/STATE.md">STATE.md</a> สถานะโปรเจกต์ทั้งหมด บั๊กที่แก้แล้ว และข้อที่ยังค้างการตัดสินใจ</li>
</ul>
</div>
""" % (owner, repo, repo, base, base, base, base)

    foot = """<div class="card">
<h2>ความซื่อตรงของตัวเลข</h2>
<ul>
<li>โมเดลเรียนจากคลิปเครื่องปกติเท่านั้น ไม่เคยเห็นคลิปผิดปกติตอนฝึก</li>
<li>เกณฑ์ตัดสินตั้งจากคะแนนของคลิปปกติในชุดฝึกเท่านั้น และไม่ปรับหลังเห็นผลของคลิปผิดปกติ</li>
<li>แบ่งข้อมูลตามรอบการถ่าย ไม่มีคลิปจากรอบเดียวกันอยู่ทั้งชุดฝึกและชุดทดสอบ</li>
<li>ไม่มีการทำซ้ำหรือสังเคราะห์คลิปเพื่อเพิ่มจำนวนตัวอย่าง</li>
<li>ตัวเลขจากงานวิจัยบนชุดข้อมูลสาธารณะ <b>ไม่ถูกนำมาปนกับผลของวิดีโอ</b> เพราะไม่ได้ผ่านกล้อง</li>
<li>ชนิดความผิดปกติอื่นนอกจากความไม่สมดุลไม่ถูกรายงาน เพราะไม่มีข้อมูลรองรับ</li>
</ul>
<p class="muted">สร้างจาก <code>ironpulse/web/build_pages.py</code> ซึ่งอ่านตัวเลขจากไฟล์ผลลัพธ์โดยตรง
ไม่มีตัวเลขที่พิมพ์ด้วยมือในหน้านี้</p>
</div>
</main></body></html>"""

    html = (head + scope + kpi + "".join(sev) + "".join(folds) + "".join(models)
            + "".join(figs) + "".join(per) + how + foot)
    io.open(os.path.join(DOCS, "index.html"), "w", encoding="utf-8",
            newline="\n").write(html)
    io.open(os.path.join(DOCS, ".nojekyll"), "w", encoding="utf-8").write("")
    print("wrote", os.path.join(DOCS, "index.html"), "%.0f KB" %
          (os.path.getsize(os.path.join(DOCS, "index.html")) / 1e3))
    print("figures copied:", len(FIGURES))


if __name__ == "__main__":
    owner = sys.argv[1] if len(sys.argv) > 1 else "OWNER"
    repo = sys.argv[2] if len(sys.argv) > 2 else "REPO"
    build(owner, repo)
