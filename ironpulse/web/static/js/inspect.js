let LAST_PATH = null;
let LAST_RESULT = null;

async function refreshMachines() {
  const list = await loadMachines(el("machine"), true);
  const opt = el("machine").selectedOptions[0];
  const note = el("machineNote");
  if (!list.length) {
    note.innerHTML = '<span class="warn" style="display:block">ยังไม่มีเครื่องในระบบ ' +
      'ให้ไปหน้าสอบเทียบเครื่องเพื่อสร้างเครื่องและ baseline ก่อน</span>';
  } else if (opt && !opt.dataset.hasBaseline) {
    note.innerHTML = '<span class="warn" style="display:block">เครื่องนี้ยังไม่มี baseline ' +
      'ให้ไปหน้าสอบเทียบเครื่องแล้วอัปโหลดคลิปสภาวะปกติก่อน จึงจะตรวจได้</span>';
  } else {
    note.textContent = "";
  }
}

async function refreshDemo() {
  const j = await API.get("/api/demo");
  const box = el("demo");
  box.innerHTML = "";
  show(el("demoEmpty"), j.files.length === 0);
  j.files.forEach(f => {
    const b = document.createElement("button");
    b.className = "chip";
    b.textContent = f.name + "  " + f.mb + " MB";
    b.addEventListener("click", () => analyse(f.path));
    box.appendChild(b);
  });
}

function drawPlots(r) {
  const grid = r.result.spectrum_grid;
  const specDraw = () => MiniPlot.draw(el("spec"), {
    logY: true,
    xLabel: "order (เท่าของรอบหมุน)",
    yLabel: "แอมพลิจูด (px)",
    series: [
      { x: grid, y: r.result.baseline_spectrum, color: "#8a97a6", width: 1.6,
        alpha: 0.75, label: "baseline" },
      { x: grid, y: r.result.spectrum, color: "#0b5fa5", width: 2.0, label: "คลิปนี้" }
    ],
    vlines: [
      { at: 1, label: "1×", color: "#14181d" },
      { at: 2, label: "2×", color: "#6b7683" },
      { at: 3, label: "3×", color: "#6b7683" }
    ]
  });
  const t = r.result.wave.map((_, i) => i * r.result.wave_dt);
  const waveDraw = () => MiniPlot.draw(el("wave"), {
    xLabel: "เวลา (วินาที)",
    yLabel: "การกระจัด (px)",
    series: [{ x: t, y: r.result.wave, color: "#0b5fa5", width: 1.2 }]
  });
  specDraw(); waveDraw();
  RESIZE = [specDraw, waveDraw];
}

function render(r) {
  LAST_RESULT = r;
  show(el("result"), true);
  const bad = r.verdict !== "ปกติ";
  const v = el("verdict");
  v.className = "verdict " + (bad ? "bad" : "ok");
  el("verdictText").textContent = r.verdict;
  el("verdictSub").textContent = bad
    ? "พบลักษณะของความไม่สมดุลของมวลหมุน"
    : "ไม่พบความผิดปกติเทียบกับ baseline";
  el("verdictMeta").textContent =
    r.filename + " · " + r.n_frames + " เฟรม · ใช้เวลา " + fmt(r.seconds, 1) + " วินาที" +
    " · " + r.detector_type + " · แกน " + r.primary_axis.replace("d_machine_", "");

  el("warnings").innerHTML = warningsHtml(r.warnings);

  const u = r.result.units;
  const cards = [
    ["คะแนน", fmt(r.score, 4), "เกณฑ์ " + fmt(r.threshold, 4)],
    ["คะแนนต่อเกณฑ์", fmt(r.ratio, 3), bad ? "เกินเกณฑ์" : "ต่ำกว่าเกณฑ์"],
    ["แอมพลิจูดที่ 1×", u.has_scale ? fmt(u.a1_um, 1) : fmt(r.a1_px, 4),
      u.has_scale ? "µm  (" + fmt(r.a1_px, 4) + " px)" : "px — ยังไม่ได้สอบเทียบสเกล"],
    ["พื้นการวัด", u.has_scale ? fmt(u.floor_um, 1) : fmt(u.floor_px, 3),
      u.has_scale ? "µm ที่ flow " + fmt(u.floor_px, 3) + " px" : "px"],
    ["ความเร็วรอบ", fmt(r.f0_hz, 3), "Hz  (" + fmt(r.f0_hz * 60, 0) + " rpm)"],
    ["baseline", r.result.baseline_clips + " คลิป", "f0 " + fmt(r.result.baseline_f0, 3) + " Hz"]
  ];
  el("stats").innerHTML = cards.map(c =>
    '<div class="stat"><div class="k">' + escapeHtml(c[0]) + '</div><div class="v">' +
    escapeHtml(c[1]) + '</div><div class="u">' + escapeHtml(c[2]) + "</div></div>").join("");

  const pct = Math.max(0, Math.min(100, r.ratio / 2 * 100));
  const bar = el("scoreBar");
  bar.className = "bar" + (bad ? " over" : "");
  bar.querySelector("i").style.width = pct + "%";
  bar.querySelector(".mark").style.left = "50%";
  el("scoreNote").textContent =
    "ขีดดำคือเกณฑ์ ซึ่งตั้งจากคะแนนของคลิปปกติในชุด baseline เท่านั้น " +
    "ชนิดความผิดปกติที่ระบบนี้รองรับมีชนิดเดียวคือ " + r.result.fault_type;

  el("waveNote").textContent =
    "แกน " + r.primary_axis.replace("d_machine_", "") + " หลังหักล้างจุดอ้างอิงแล้ว · " +
    "แสดงทุก " + fmt(r.result.wave_dt * 1000, 1) + " มิลลิวินาที";

  drawPlots(r);
}

async function analyse(path) {
  const mid = el("machine").value;
  el("errors").innerHTML = "";
  if (!mid) { errorBox(el("errors"), "ยังไม่ได้เลือกเครื่อง ให้สร้างเครื่องที่หน้าสอบเทียบก่อน"); return; }
  LAST_PATH = path;
  show(el("result"), false);
  show(el("progress"), true);
  el("progressText").textContent = "กำลังวิเคราะห์ " + path.split(/[\\/]/).pop();
  el("progressBar").style.width = "0%";
  try {
    const form = new FormData();
    form.append("machine_id", mid);
    form.append("path", path);
    form.append("fs", el("fs").value);
    const j = await API.post("/api/inspect", form);
    const res = await pollJob(j.job, t => {
      if (t.frames_total) {
        const p = t.frames / t.frames_total * 100;
        el("progressBar").style.width = p.toFixed(1) + "%";
        el("progressText").textContent =
          "ติดตามแล้ว " + t.frames + " จาก " + t.frames_total + " เฟรม";
      }
    });
    show(el("progress"), false);
    render(res);
  } catch (e) {
    show(el("progress"), false);
    errorBox(el("errors"), e.message);
  }
}

async function uploadAndAnalyse(files) {
  if (!files || !files.length) return;
  el("errors").innerHTML = "";
  try {
    const form = new FormData();
    for (const f of files) form.append("files", f);
    const j = await API.post("/api/upload", form);
    await analyse(j.files[0].path);
  } catch (e) {
    errorBox(el("errors"), e.message);
  }
}

const drop = el("drop");
["dragenter", "dragover"].forEach(t =>
  drop.addEventListener(t, e => { e.preventDefault(); drop.classList.add("hot"); }));
["dragleave", "drop"].forEach(t =>
  drop.addEventListener(t, e => { e.preventDefault(); drop.classList.remove("hot"); }));
drop.addEventListener("drop", e => uploadAndAnalyse(e.dataTransfer.files));
el("pick").addEventListener("click", () => el("file").click());
el("file").addEventListener("change", e => uploadAndAnalyse(e.target.files));
el("again").addEventListener("click", () => { if (LAST_PATH) analyse(LAST_PATH); });
el("machine").addEventListener("change", refreshMachines);

const reseedModal = el("reseedModal");
let picker = null;
el("reseed").addEventListener("click", async () => {
  show(reseedModal, true);
  const j = await API.get("/api/demo");
  const sel = el("reseedFile");
  sel.innerHTML = "";
  j.files.forEach(f => {
    const o = document.createElement("option");
    o.value = f.path; o.textContent = f.name;
    sel.appendChild(o);
  });
  picker = makePicker(el("reseedPicker"), st => {
    el("reseedSave").disabled = st.points.length !== 3;
  });
});
el("reseedCancel").addEventListener("click", () => show(reseedModal, false));
el("reseedLoad").addEventListener("click", async () => {
  try { await picker.load(el("reseedFile").value); }
  catch (e) { alert(e.message); }
});
el("reseedSave").addEventListener("click", async () => {
  try {
    const form = new FormData();
    form.append("roi", JSON.stringify(picker.state.points));
    form.append("fs", el("fs").value);
    await API.post("/api/machine/" + el("machine").value + "/roi", form);
    show(reseedModal, false);
    await refreshMachines();
  } catch (e) { alert(e.message); }
});

refreshMachines();
refreshDemo();
