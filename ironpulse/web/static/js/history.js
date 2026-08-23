let ROWS = [];

async function load() {
  const mid = el("machine").value;
  const wrap = el("tableWrap");
  if (!mid) { wrap.innerHTML = '<div class="muted">ยังไม่มีเครื่องในระบบ</div>'; return; }
  const j = await API.get("/api/history/" + mid);
  ROWS = j.rows.slice().reverse();
  if (!ROWS.length) {
    wrap.innerHTML = '<div class="muted">ยังไม่มีการตรวจของเครื่องนี้ ' +
      'ให้ไปหน้าตรวจวินิจฉัยแล้ววิเคราะห์คลิปสักไฟล์</div>';
    MiniPlot.draw(el("trendScore"), { series: [] });
    MiniPlot.draw(el("trendA1"), { series: [] });
    return;
  }
  const hasScale = ROWS.some(r => r.a1_um !== null);
  el("a1Note").textContent = hasScale
    ? "แอมพลิจูดที่ 1× เป็นไมโครเมตร"
    : "ยังไม่มีค่าสอบเทียบสเกล จึงแสดงเป็นพิกเซล";

  wrap.innerHTML =
    "<table><thead><tr><th>เวลา</th><th>ไฟล์</th><th>ผล</th><th>คะแนน</th>" +
    "<th>เกณฑ์</th><th>คะแนน/เกณฑ์</th><th>1× " + (hasScale ? "µm" : "px") + "</th>" +
    "<th>f0 Hz</th><th>วินาที</th><th>คำเตือน</th></tr></thead><tbody>" +
    ROWS.map(r =>
      '<tr class="click" data-id="' + r.id + '"><td>' + escapeHtml(r.created_at) +
      "</td><td>" + escapeHtml(r.filename) +
      '</td><td class="' + (r.verdict === "ปกติ" ? "ok" : "bad") + '">' +
      escapeHtml(r.verdict) + "</td><td>" + fmt(r.score, 4) + "</td><td>" +
      fmt(r.threshold, 4) + "</td><td>" + fmt(r.ratio, 3) + "</td><td>" +
      (hasScale ? fmt(r.a1_um, 1) : fmt(r.a1_px, 4)) + "</td><td>" +
      fmt(r.f0_hz, 3) + "</td><td>" + fmt(r.seconds, 1) + "</td><td>" +
      (r.n_warnings ? r.n_warnings + " ข้อ" : "ไม่มี") + "</td></tr>").join("") +
    "</tbody></table>";
  wrap.querySelectorAll("tr.click").forEach(tr =>
    tr.addEventListener("click", () => openDetail(tr.dataset.id)));

  const x = ROWS.map((_, i) => i + 1);
  const thr = ROWS.length ? ROWS[ROWS.length - 1].threshold : 0;
  const drawScore = () => MiniPlot.draw(el("trendScore"), {
    xLabel: "ลำดับการตรวจ", yLabel: "คะแนน",
    series: [{ x: x, y: ROWS.map(r => r.score), color: "#0b5fa5", width: 2, dots: 3 }],
    hlines: [{ at: thr, color: "#b02a12" }]
  });
  const drawA1 = () => MiniPlot.draw(el("trendA1"), {
    xLabel: "ลำดับการตรวจ", yLabel: hasScale ? "1× (µm)" : "1× (px)",
    series: [{ x: x, y: ROWS.map(r => hasScale ? r.a1_um : r.a1_px),
               color: "#D55E00", width: 2, dots: 3 }],
    hlines: hasScale && ROWS[0].floor_um ? [{ at: ROWS[0].floor_um, color: "#5a6572" }] : []
  });
  drawScore(); drawA1();
  RESIZE = [drawScore, drawA1];
}

async function openDetail(id) {
  const r = await API.get("/api/inspection/" + id);
  show(el("detail"), true);
  el("detailTitle").textContent = r.filename;
  const bad = r.verdict !== "ปกติ";
  el("detailVerdict").innerHTML =
    '<div class="verdict ' + (bad ? "bad" : "ok") + '"><div class="big" style="font-size:44px">' +
    escapeHtml(r.verdict) + "</div><div class='sub'>" +
    escapeHtml(r.created_at) + "</div></div>";
  el("detailWarn").innerHTML = warningsHtml(r.warnings);
  const u = r.result.units;
  const cards = [
    ["คะแนน", fmt(r.score, 4), "เกณฑ์ " + fmt(r.threshold, 4)],
    ["คะแนน/เกณฑ์", fmt(r.ratio, 3), ""],
    ["1×", u.has_scale ? fmt(r.a1_um, 1) + " µm" : fmt(r.a1_px, 4) + " px",
      u.has_scale ? "พื้นการวัด " + fmt(r.floor_um, 1) + " µm" : "ไม่มีค่าสอบเทียบ"],
    ["ความเร็วรอบ", fmt(r.f0_hz, 3), "Hz"],
    ["เฟรม", r.n_frames, "ใช้เวลา " + fmt(r.seconds, 1) + " วินาที"]
  ];
  el("detailStats").innerHTML = cards.map(c =>
    '<div class="stat"><div class="k">' + escapeHtml(c[0]) + '</div><div class="v">' +
    escapeHtml(String(c[1])) + '</div><div class="u">' + escapeHtml(c[2]) + "</div></div>").join("");
  const grid = r.result.spectrum_grid;
  MiniPlot.draw(el("detailSpec"), {
    logY: true, xLabel: "order", yLabel: "แอมพลิจูด (px)",
    series: [
      { x: grid, y: r.result.baseline_spectrum, color: "#8a97a6", alpha: 0.75, label: "baseline" },
      { x: grid, y: r.result.spectrum, color: "#0b5fa5", width: 2, label: "คลิปนี้" }],
    vlines: [{ at: 1, label: "1×", color: "#14181d" }, { at: 2, label: "2×", color: "#6b7683" },
             { at: 3, label: "3×", color: "#6b7683" }]
  });
  const t = r.result.wave.map((_, i) => i * r.result.wave_dt);
  MiniPlot.draw(el("detailWave"), {
    xLabel: "เวลา (วินาที)", yLabel: "การกระจัด (px)",
    series: [{ x: t, y: r.result.wave, color: "#0b5fa5", width: 1.2 }]
  });
  el("detailProv").textContent =
    "detector " + r.detector_type + " · config " + r.config_hash +
    " · core " + r.core_version + " · fs " + r.fs +
    " · scale_id " + (r.scale_id || "ไม่มี") + " · แกน " + r.primary_axis;
}

el("detailClose").addEventListener("click", () => show(el("detail"), false));
el("machine").addEventListener("change", load);
el("export").addEventListener("click", () => {
  const mid = el("machine").value;
  if (mid) window.location = "/api/history/" + mid + "/csv";
});

(async function () {
  await loadMachines(el("machine"), false);
  await load();
})();
