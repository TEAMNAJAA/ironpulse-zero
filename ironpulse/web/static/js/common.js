const API = {
  async get(url) {
    const r = await fetch(url);
    const j = await r.json().catch(() => ({ detail: "อ่านคำตอบจากเซิร์ฟเวอร์ไม่ได้" }));
    if (!r.ok) throw new Error(j.detail || ("HTTP " + r.status));
    return j;
  },
  async post(url, form) {
    const r = await fetch(url, { method: "POST", body: form });
    const j = await r.json().catch(() => ({ detail: "อ่านคำตอบจากเซิร์ฟเวอร์ไม่ได้" }));
    if (!r.ok) throw new Error(j.detail || ("HTTP " + r.status));
    return j;
  }
};

function el(id) { return document.getElementById(id); }

function show(node, on) { node.classList.toggle("hide", !on); }

function fmt(v, d) {
  if (v === null || v === undefined || !isFinite(v)) return "—";
  return Number(v).toFixed(d === undefined ? 2 : d);
}

function errorBox(container, message) {
  container.innerHTML = '<div class="err">' + escapeHtml(message) + "</div>";
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, function (c) {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
  });
}

function warningsHtml(list) {
  if (!list || !list.length) return "";
  return list.map(w => '<div class="warn">' + escapeHtml(w) + "</div>").join("");
}

async function pollJob(jobId, onTick) {
  for (;;) {
    const j = await API.get("/api/job/" + jobId);
    if (onTick) onTick(j);
    if (j.state === "done") return j.result;
    if (j.state === "error") throw new Error(j.error);
    await new Promise(r => setTimeout(r, 350));
  }
}

async function loadMachines(selectNode, keep) {
  const j = await API.get("/api/machines");
  const cur = keep ? selectNode.value : null;
  selectNode.innerHTML = "";
  if (!j.machines.length) {
    const o = document.createElement("option");
    o.value = "";
    o.textContent = "ยังไม่มีเครื่องในระบบ";
    selectNode.appendChild(o);
  }
  j.machines.forEach(m => {
    const o = document.createElement("option");
    o.value = m.id;
    o.textContent = m.name + (m.has_baseline
      ? " · baseline " + m.baseline_clips + " คลิป"
      : " · ยังไม่มี baseline");
    o.dataset.hasBaseline = m.has_baseline ? "1" : "";
    selectNode.appendChild(o);
  });
  if (cur) selectNode.value = cur;
  return j.machines;
}

function redraw(list) {
  list.forEach(fn => fn());
}

let RESIZE = [];
window.addEventListener("resize", () => RESIZE.forEach(fn => fn()));
