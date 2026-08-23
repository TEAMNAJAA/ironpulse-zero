const MIN_CLIPS = Number(document.body.dataset.minClips || 20);
let FILES = [];
let MACHINE_ID = null;
let picker = makePicker(el("picker"), () => preflight());

async function refresh() {
  await loadMachines(el("machine"), true);
  const j = await API.get("/api/scales");
  const sel = el("scale");
  sel.innerHTML = '<option value="">ยังไม่มี — จะแสดงผลเป็นพิกเซล</option>';
  j.scales.forEach(s => {
    const o = document.createElement("option");
    o.value = s.id;
    o.textContent = s.id + "  " + s.um_per_px + " µm/px  " + s.width + "×" + s.height;
    sel.appendChild(o);
  });
}

function listFiles() {
  const box = el("fileList");
  show(box, FILES.length > 0);
  box.innerHTML = FILES.map(f =>
    "<div>" + escapeHtml(f.name) + "  <span class='muted'>" + f.mb + " MB</span></div>").join("");
  el("fileNote").textContent = FILES.length
    ? "เลือกไว้ " + FILES.length + " คลิป"
    : "ยังไม่ได้เลือกคลิป";
  preflight();
}

function preflight() {
  const box = el("preflight");
  const msgs = [];
  let ok = true;
  if (!MACHINE_ID && !el("name").value.trim()) {
    msgs.push(["err", "ยังไม่ได้เลือกเครื่องเดิมหรือกรอกชื่อเครื่องใหม่"]);
    ok = false;
  }
  if (FILES.length < MIN_CLIPS) {
    msgs.push(["err", "มีคลิป " + FILES.length + " คลิป ต้องการอย่างน้อย " + MIN_CLIPS +
      " คลิป ขาดอีก " + (MIN_CLIPS - FILES.length) + " คลิป จึงจะสร้าง baseline ได้"]);
    ok = false;
  }
  if (picker.state.points.length !== 3) {
    msgs.push(["err", "ยังคลิกมาร์กเกอร์ไม่ครบ 3 จุด (คลิกแล้ว " +
      picker.state.points.length + " จุด)"]);
    ok = false;
  }
  if (picker.state.meta_fps && el("fs").value) {
    const meta = Number(picker.state.meta_fps), fs = Number(el("fs").value);
    if (meta > 0 && Math.abs(meta - fs) / fs > 0.1) {
      msgs.push(["warn", "metadata ของไฟล์บอก " + meta + " fps แต่คุณกรอก " + fs +
        " Hz ต่างกันเกิน 10% ระบบจะใช้ค่าที่คุณกรอก " +
        "ถ้าถ่ายสโลว์โมชั่นค่านี้ถูกแล้ว ถ้าไม่ใช่ให้แก้ก่อนสร้าง baseline"]);
    }
  }
  if (ok) msgs.push(["ok-note", "พร้อมสร้าง baseline จาก " + FILES.length + " คลิป"]);
  box.innerHTML = msgs.map(m => '<div class="' + m[0] + '">' + escapeHtml(m[1]) + "</div>").join("");
  el("build").disabled = !ok;
  return ok;
}

async function addFiles(fileObjs) {
  el("out").innerHTML = "";
  try {
    const form = new FormData();
    for (const f of fileObjs) form.append("files", f);
    const j = await API.post("/api/upload", form);
    FILES = FILES.concat(j.files);
    listFiles();
    if (FILES.length && !picker.state.url) await picker.load(FILES[0].path);
  } catch (e) {
    errorBox(el("out"), e.message);
  }
}

const drop = el("drop");
["dragenter", "dragover"].forEach(t =>
  drop.addEventListener(t, e => { e.preventDefault(); drop.classList.add("hot"); }));
["dragleave", "drop"].forEach(t =>
  drop.addEventListener(t, e => { e.preventDefault(); drop.classList.remove("hot"); }));
drop.addEventListener("drop", e => addFiles(e.dataTransfer.files));
el("pick").addEventListener("click", () => el("file").click());
el("file").addEventListener("change", e => addFiles(e.target.files));
el("clearFiles").addEventListener("click", () => { FILES = []; listFiles(); });

el("useDemo").addEventListener("click", async () => {
  const j = await API.get("/api/demo");
  FILES = j.files.map(f => ({ name: f.name, path: f.path, mb: f.mb }));
  listFiles();
  if (FILES.length) await picker.load(FILES[0].path);
});

el("useExisting").addEventListener("click", async () => {
  const sel = el("machine");
  if (!sel.value) return;
  MACHINE_ID = Number(sel.value);
  const list = await API.get("/api/machines");
  const m = list.machines.find(x => x.id === MACHINE_ID);
  if (m) {
    el("name").value = m.name;
    el("blades").value = m.blade_count || "";
    el("fs").value = m.fs;
    el("scale").value = m.scale_id || "";
    if (picker.state.url) picker.set(m.roi);
  }
  preflight();
});

["name", "blades", "fs"].forEach(id => el(id).addEventListener("input", preflight));

el("build").addEventListener("click", async () => {
  if (!preflight()) return;
  el("out").innerHTML = "";
  el("build").disabled = true;
  show(el("progress"), true);
  try {
    if (!MACHINE_ID) {
      const form = new FormData();
      form.append("name", el("name").value.trim());
      form.append("blade_count", el("blades").value);
      form.append("fs", el("fs").value);
      form.append("scale_id", el("scale").value);
      form.append("roi", JSON.stringify(picker.state.points));
      const j = await API.post("/api/machines", form);
      MACHINE_ID = j.machine_id;
    } else {
      const form = new FormData();
      form.append("roi", JSON.stringify(picker.state.points));
      form.append("fs", el("fs").value);
      form.append("blade_count", el("blades").value);
      form.append("scale_id", el("scale").value);
      await API.post("/api/machine/" + MACHINE_ID + "/roi", form);
    }
    const form2 = new FormData();
    form2.append("machine_id", MACHINE_ID);
    form2.append("paths", JSON.stringify(FILES.map(f => f.path)));
    const j2 = await API.post("/api/baseline", form2);
    const res = await pollJob(j2.job, t => {
      const total = t.step_total || 1;
      const inner = t.frames_total ? t.frames / t.frames_total : 0;
      const p = Math.min(100, (t.step + inner) / total * 100);
      el("progressBar").style.width = p.toFixed(1) + "%";
      el("progressText").textContent =
        "คลิปที่ " + Math.min(t.step + 1, total) + " จาก " + total +
        (t.message ? " · " + t.message : "") +
        (t.frames_total ? " · ติดตามแล้ว " + t.frames + "/" + t.frames_total + " เฟรม" : "");
    });
    show(el("progress"), false);
    el("out").innerHTML =
      '<div class="ok-note">สร้าง baseline สำเร็จ · ' + res.n_clips + ' คลิป · ' +
      res.detector + ' · ชุด feature ' + res.feature_set + ' จำนวน ' + res.n_features +
      ' ตัว · แกน ' + res.primary_axis.replace("d_machine_", "") +
      ' · f0 ' + fmt(res.f0_hz, 3) + ' Hz · เกณฑ์ ' + fmt(res.threshold, 4) +
      ' · รอบการถ่ายที่ตรวจพบ ' + res.sessions.join(", ") + "</div>" +
      warningsHtml(res.warnings) +
      '<div class="muted">ไปหน้าตรวจวินิจฉัยเพื่อเริ่มใช้งานได้เลย</div>';
    await refresh();
  } catch (e) {
    show(el("progress"), false);
    errorBox(el("out"), e.message);
  }
  el("build").disabled = false;
});

refresh();
listFiles();
