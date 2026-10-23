// Expansion and product widgets for lesson 7. Needs decoder.js first (window.aaqDecode).
//
// <div class="decoder" data-expand data-a="1D,4,128" data-b="44,4,130"></div>
//   Two scaled operands (byte, fe, scale s): each expanded value, the exact product,
//   the FP32-rounded product (MULT_RES holds FP32), and whether the two agree.
// <div class="decoder" data-window data-scales="128,128,130,126" data-offset="448"></div>
//   R_CYCLIC as 4 quarters of 128 elements, each with its own row scale; shows which
//   quarters a 128-element MULT.RC.* window starting at `offset` reads, and the scale
//   each part of the window uses under the per-element rule.
(function () {
  function num(v) {
    if (!isFinite(v)) return v > 0 ? "+inf" : (v < 0 ? "−inf" : "NaN");
    return v === 0 ? "0" : (Math.abs(v) < 1e-4 || Math.abs(v) >= 1e7 ? v.toExponential(6) : String(+v.toPrecision(8)));
  }
  function field(uid, id, label, value, attrs) {
    return '<label for="' + uid + id + '">' + label + '<input id="' + uid + id + '" value="' + value + '" ' + (attrs || "") + "></label>";
  }

  function buildExpand(host) {
    var uid = "x" + Math.random().toString(36).slice(2, 7);
    var a = (host.dataset.a || "1D,4,128").split(","), b = (host.dataset.b || "44,4,130").split(",");
    host.innerHTML =
      '<div class="controls">' +
      field(uid, "ab", "A byte (hex)", a[0], 'maxlength="2" autocomplete="off"') +
      field(uid, "af", "A fe", a[1], 'type="number" min="1" max="6"') +
      field(uid, "as", "A scale s", a[2], 'type="number" min="0" max="254"') +
      "</div>" +
      '<div class="controls">' +
      field(uid, "bb", "B byte (hex)", b[0], 'maxlength="2" autocomplete="off"') +
      field(uid, "bf", "B fe", b[1], 'type="number" min="1" max="6"') +
      field(uid, "bs", "B scale s", b[2], 'type="number" min="0" max="254"') +
      "</div>" +
      '<div class="table-wrap"><table><tbody></tbody></table></div>';
    var ids = ["ab", "af", "as", "bb", "bf", "bs"].map(function (k) { return host.querySelector("#" + uid + k); });
    var tbody = host.querySelector("tbody");
    function operand(i) {
      var byte = parseInt(ids[i].value, 16), fe = +ids[i + 1].value, s = +ids[i + 2].value;
      if (isNaN(byte) || byte < 0 || byte > 255 || !(fe >= 1 && fe <= 6) || !(s >= 0 && s <= 254)) return null;
      return window.aaqDecode(byte, fe, s);
    }
    function render() {
      var A = operand(0), B = operand(3);
      if (!A || !B) { tbody.innerHTML = '<tr><td>Enter a byte 00–FF, fe 1–6 and a scale 0–254 for both operands.</td></tr>'; return; }
      var exact = A.value * B.value; // exact in a double: two ≤7-bit significands, exponents far inside double range
      var f32 = Math.fround(exact);
      var verdict = exact === f32 ? '<span class="role aaq">exact in FP32</span>'
        : (!isFinite(f32) ? '<span class="role meta">overflows FP32</span>' : '<span class="role meta">rounded or flushed in FP32</span>');
      tbody.innerHTML =
        '<tr><th>A expanded</th><td class="num">' + num(A.value) + '</td><td class="mono">' + A.formula + "</td></tr>" +
        '<tr><th>B expanded</th><td class="num">' + num(B.value) + '</td><td class="mono">' + B.formula + "</td></tr>" +
        '<tr><th>Exact product</th><td class="num">' + num(exact) + "</td><td></td></tr>" +
        '<tr><th>MULT_RES (FP32)</th><td class="num">' + num(f32) + "</td><td>" + verdict + "</td></tr>";
    }
    ids.forEach(function (el) { el.addEventListener("input", render); });
    render();
  }

  // Which parts of a 128-element window starting at `offset` come from which quarter.
  function windowParts(offset) {
    var parts = [], start = ((offset % 512) + 512) % 512, i = 0;
    while (i < 128) {
      var pos = (start + i) % 512, q = Math.floor(pos / 128), run = Math.min(128 - (pos % 128), 128 - i);
      parts.push({ quarter: q, from: i, to: i + run - 1, ringFrom: pos, ringTo: pos + run - 1 });
      i += run;
    }
    return parts;
  }

  function buildWindow(host) {
    var uid = "w" + Math.random().toString(36).slice(2, 7);
    var scales = (host.dataset.scales || "128,128,130,126").split(",");
    var html = '<div class="controls">';
    for (var q = 0; q < 4; q++) html += field(uid, "q" + q, "Quarter " + q + " scale", scales[q], 'type="number" min="0" max="254"');
    html += field(uid, "o", "Window offset (0–511)", host.dataset.offset || "448", 'type="number" min="0" max="511"') + "</div>" +
      '<div class="cells" aria-hidden="true" style="grid-template-columns:repeat(64,minmax(0,1fr))"></div>' +
      '<p class="bitlegend">R_CYCLIC as 64 cells of 8 elements; the window is filled. Quarters alternate blue and amber.</p>' +
      '<p class="result" aria-live="polite"></p>';
    host.innerHTML = html;
    var qIn = [0, 1, 2, 3].map(function (k) { return host.querySelector("#" + uid + "q" + k); });
    var oIn = host.querySelector("#" + uid + "o"), cells = host.querySelector(".cells"), out = host.querySelector(".result");
    function render() {
      var off = +oIn.value, sc = qIn.map(function (el) { return +el.value; });
      if (!(off >= 0 && off <= 511)) { out.textContent = "Enter an offset from 0 to 511."; return; }
      var parts = windowParts(off), inWin = {};
      for (var i = 0; i < 128; i++) inWin[Math.floor(((off + i) % 512) / 8)] = true;
      var c = "";
      for (var k = 0; k < 64; k++) {
        var q = Math.floor(k / 16), even = q % 2 === 0;
        var border = even ? "var(--data)" : "var(--meta)", fill = inWin[k] ? border : (even ? "var(--data-tint)" : "var(--meta-tint)");
        c += '<span style="border-color:' + border + ";background:" + fill + '"></span>';
      }
      cells.innerHTML = c;
      var lines = parts.map(function (p) {
        return "window elements " + p.from + "–" + p.to + " ← quarter " + p.quarter + " (ring " + p.ringFrom + "–" + p.ringTo + "), scale " + sc[p.quarter];
      });
      var distinct = parts.map(function (p) { return sc[p.quarter]; }).filter(function (v, i, arr) { return arr.indexOf(v) === i; });
      lines.push(parts.length === 1 ? "One quarter: one scale for the whole window."
        : (distinct.length === 1 ? "Two quarters with equal scales: no conflict."
          : "Two quarters with different scales: per-element rule needed, or a fault, or one scale wins."));
      out.innerHTML = lines.join("<br>");
    }
    qIn.concat([oIn]).forEach(function (el) { el.addEventListener("input", render); });
    render();
  }

  window.aaqWindowParts = windowParts;
  function init() {
    document.querySelectorAll("[data-expand]").forEach(buildExpand);
    document.querySelectorAll("[data-window]").forEach(buildWindow);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
