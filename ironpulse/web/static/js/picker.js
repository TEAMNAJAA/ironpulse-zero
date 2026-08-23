const MARKER_LABELS = [
  { n: "1", t: "จุดบนเครื่อง — ติดบนตัวเรือนมอเตอร์ ส่วนที่สั่นมากที่สุด" },
  { n: "2", t: "จุดบนเสา — ติดบนฐานหรือเสาของเครื่อง ใช้จับการหลวมของจุดยึด" },
  { n: "3", t: "จุดอ้างอิง — ติดบนผนังหรือของที่ไม่สั่น ใช้หักล้างการสั่นของกล้อง" }
];

function makePicker(container, onChange) {
  const state = { points: [], url: null, w: 0, h: 0, path: null };
  container.innerHTML =
    '<div class="muted" id="pkHint"></div>' +
    '<div class="framewrap" id="pkWrap"><img id="pkImg" alt=""></div>' +
    '<div class="row" style="margin-top:8px">' +
    '<button class="small quiet" id="pkUndo">ย้อนกลับหนึ่งจุด</button>' +
    '<button class="small quiet" id="pkClear">ล้างทั้งหมด</button></div>';

  const wrap = container.querySelector("#pkWrap");
  const img = container.querySelector("#pkImg");
  const hint = container.querySelector("#pkHint");

  function paint() {
    wrap.querySelectorAll(".pin").forEach(n => n.remove());
    const rect = img.getBoundingClientRect();
    const sx = rect.width / state.w, sy = rect.height / state.h;
    state.points.forEach((p, i) => {
      const d = document.createElement("div");
      d.className = "pin p" + i;
      d.style.left = (p[0] * sx) + "px";
      d.style.top = (p[1] * sy) + "px";
      d.textContent = i + 1;
      wrap.appendChild(d);
    });
    const i = state.points.length;
    hint.innerHTML = i < 3
      ? "คลิกจุดที่ " + MARKER_LABELS[i].n + " — " + escapeHtml(MARKER_LABELS[i].t)
      : "ครบ 3 จุดแล้ว · " + state.points.map(p => "(" + p[0] + ", " + p[1] + ")").join("  ");
    if (onChange) onChange(state);
  }

  img.addEventListener("click", ev => {
    if (!state.url || state.points.length >= 3) return;
    const rect = img.getBoundingClientRect();
    const x = Math.round((ev.clientX - rect.left) / rect.width * state.w);
    const y = Math.round((ev.clientY - rect.top) / rect.height * state.h);
    state.points.push([x, y]);
    paint();
  });
  container.querySelector("#pkUndo").addEventListener("click", () => {
    state.points.pop(); paint();
  });
  container.querySelector("#pkClear").addEventListener("click", () => {
    state.points = []; paint();
  });
  window.addEventListener("resize", paint);

  return {
    state,
    async load(path) {
      const form = new FormData();
      form.append("path", path);
      const j = await API.post("/api/first-frame", form);
      state.url = j.url; state.w = j.width; state.h = j.height;
      state.path = j.path; state.points = [];
      state.n_frames = j.n_frames; state.meta_fps = j.meta_fps;
      img.src = j.url;
      await new Promise(r => { img.onload = r; });
      paint();
      return j;
    },
    set(points) { state.points = points.slice(0, 3); paint(); }
  };
}
