(function (global) {
  "use strict";

  function dpr() { return global.devicePixelRatio || 1; }

  function fit(canvas) {
    var r = dpr();
    var w = canvas.clientWidth || 600;
    var h = canvas.clientHeight || 260;
    if (canvas.width !== Math.round(w * r) || canvas.height !== Math.round(h * r)) {
      canvas.width = Math.round(w * r);
      canvas.height = Math.round(h * r);
    }
    var ctx = canvas.getContext("2d");
    ctx.setTransform(r, 0, 0, r, 0, 0);
    return { ctx: ctx, w: w, h: h };
  }

  function niceTicks(lo, hi, n) {
    if (!isFinite(lo) || !isFinite(hi) || hi <= lo) return [lo];
    var raw = (hi - lo) / Math.max(1, n);
    var mag = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10));
    var norm = raw / mag;
    var step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
    var out = [], v = Math.ceil(lo / step) * step;
    for (; v <= hi + step * 1e-6; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
    return out;
  }

  function logTicks(lo, hi) {
    var out = [];
    var a = Math.floor(Math.log(lo) / Math.LN10);
    var b = Math.ceil(Math.log(hi) / Math.LN10);
    for (var e = a; e <= b; e++) {
      var v = Math.pow(10, e);
      if (v >= lo && v <= hi) out.push(v);
    }
    return out;
  }

  function fmt(v) {
    if (v === 0) return "0";
    var a = Math.abs(v);
    if (a >= 1e4 || a < 1e-3) return v.toExponential(0);
    if (a >= 100) return v.toFixed(0);
    if (a >= 10) return v.toFixed(1);
    if (a >= 1) return v.toFixed(2);
    return v.toFixed(3);
  }

  function extent(series, key) {
    var lo = Infinity, hi = -Infinity;
    series.forEach(function (s) {
      for (var i = 0; i < s[key].length; i++) {
        var v = s[key][i];
        if (isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
      }
    });
    return [lo, hi];
  }

  function draw(canvas, opts) {
    var f = fit(canvas), ctx = f.ctx, W = f.w, H = f.h;
    var css = getComputedStyle(document.documentElement);
    var ink = css.getPropertyValue("--ink").trim() || "#111";
    var grid = css.getPropertyValue("--grid").trim() || "#ddd";
    var pad = opts.pad || { l: 62, r: 14, t: 12, b: 34 };
    ctx.clearRect(0, 0, W, H);

    var series = (opts.series || []).filter(function (s) { return s.x && s.x.length; });
    if (!series.length) return;

    var xe = opts.xRange || extent(series, "x");
    var ye = opts.yRange || extent(series, "y");
    var logy = !!opts.logY;
    if (logy) {
      var pos = [];
      series.forEach(function (s) {
        s.y.forEach(function (v) { if (v > 0 && isFinite(v)) pos.push(v); });
      });
      pos.sort(function (a, b) { return a - b; });
      ye = [pos.length ? pos[Math.floor(pos.length * 0.002)] : 1e-6,
            pos.length ? pos[pos.length - 1] : 1];
      if (ye[0] <= 0) ye[0] = 1e-7;
      ye[1] *= 1.6;
    } else {
      var span = ye[1] - ye[0];
      if (!isFinite(span) || span === 0) span = Math.abs(ye[1]) || 1;
      ye = [ye[0] - span * 0.08, ye[1] + span * 0.12];
    }

    var pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
    function px(v) { return pad.l + (v - xe[0]) / (xe[1] - xe[0] || 1) * pw; }
    function py(v) {
      if (logy) {
        var a = Math.log(Math.max(v, ye[0])) / Math.LN10;
        var lo = Math.log(ye[0]) / Math.LN10, hi = Math.log(ye[1]) / Math.LN10;
        return pad.t + ph - (a - lo) / (hi - lo || 1) * ph;
      }
      return pad.t + ph - (v - ye[0]) / (ye[1] - ye[0] || 1) * ph;
    }

    ctx.font = "12px system-ui, sans-serif";
    ctx.strokeStyle = grid;
    ctx.fillStyle = ink;
    ctx.lineWidth = 1;

    var yt = logy ? logTicks(ye[0], ye[1]) : niceTicks(ye[0], ye[1], 5);
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    yt.forEach(function (v) {
      var y = py(v);
      ctx.globalAlpha = 0.45;
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.fillText(fmt(v), pad.l - 8, y);
    });

    var xt = niceTicks(xe[0], xe[1], 6);
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    xt.forEach(function (v) {
      var x = px(v);
      ctx.globalAlpha = 0.45;
      ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, pad.t + ph); ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.fillText(fmt(v), x, pad.t + ph + 6);
    });

    (opts.vlines || []).forEach(function (v) {
      var x = px(v.at);
      ctx.strokeStyle = v.color || "#888";
      ctx.setLineDash(v.dash || [5, 4]);
      ctx.lineWidth = v.width || 1.5;
      ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, pad.t + ph); ctx.stroke();
      ctx.setLineDash([]);
      if (v.label) {
        ctx.fillStyle = v.color || "#888";
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillText(v.label, x, pad.t + 2);
        ctx.fillStyle = ink;
      }
    });

    (opts.hlines || []).forEach(function (v) {
      var y = py(v.at);
      ctx.strokeStyle = v.color || "#888";
      ctx.setLineDash(v.dash || [6, 4]);
      ctx.lineWidth = v.width || 2;
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
      ctx.setLineDash([]);
    });

    series.forEach(function (s) {
      ctx.strokeStyle = s.color || "#0072B2";
      ctx.lineWidth = s.width || 1.6;
      ctx.globalAlpha = s.alpha === undefined ? 1 : s.alpha;
      ctx.beginPath();
      var started = false;
      for (var i = 0; i < s.x.length; i++) {
        var yv = s.y[i];
        if (!isFinite(yv) || (logy && yv <= 0)) { started = false; continue; }
        var X = px(s.x[i]), Y = py(yv);
        if (!started) { ctx.moveTo(X, Y); started = true; } else { ctx.lineTo(X, Y); }
      }
      ctx.stroke();
      if (s.dots) {
        ctx.fillStyle = s.color || "#0072B2";
        for (var k = 0; k < s.x.length; k++) {
          if (!isFinite(s.y[k])) continue;
          ctx.beginPath();
          ctx.arc(px(s.x[k]), py(s.y[k]), s.dots, 0, Math.PI * 2);
          ctx.fill();
        }
      }
      ctx.globalAlpha = 1;
    });

    ctx.strokeStyle = ink;
    ctx.globalAlpha = 0.55;
    ctx.lineWidth = 1.2;
    ctx.strokeRect(pad.l, pad.t, pw, ph);
    ctx.globalAlpha = 1;

    if (opts.xLabel) {
      ctx.textAlign = "center"; ctx.textBaseline = "bottom";
      ctx.fillText(opts.xLabel, pad.l + pw / 2, H - 2);
    }
    if (opts.yLabel) {
      ctx.save();
      ctx.translate(12, pad.t + ph / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center"; ctx.textBaseline = "top";
      ctx.fillText(opts.yLabel, 0, 0);
      ctx.restore();
    }

    var legend = series.filter(function (s) { return s.label; });
    if (legend.length) {
      var lx = W - pad.r - 10, ly = pad.t + 12;
      ctx.textAlign = "right"; ctx.textBaseline = "middle";
      legend.forEach(function (s, i) {
        var y = ly + i * 18;
        ctx.fillStyle = ink;
        ctx.fillText(s.label, lx - 22, y);
        ctx.strokeStyle = s.color || "#0072B2";
        ctx.lineWidth = 3;
        ctx.globalAlpha = s.alpha === undefined ? 1 : s.alpha;
        ctx.beginPath(); ctx.moveTo(lx - 18, y); ctx.lineTo(lx, y); ctx.stroke();
        ctx.globalAlpha = 1;
      });
    }
  }

  global.MiniPlot = { draw: draw };
})(window);
