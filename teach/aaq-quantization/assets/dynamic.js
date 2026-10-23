// AaQ dynamic exponent-field selection: MAS-stage-aaq-str.md §5.1.1, signed.
// The decision follows §5.1.1 as written, which reproduces its worked example:
//   err = truncation error (|x| - floor(|x| / step) * step), with the §5.1.1 table's steps:
//     fe1: 2^(S - min(E, 1) - fm1)        fe2: 2^(S - min(E, 3) - fm1 + 1)      (S = s - 127)
//   A (E = 0)            = RTN(|err2^2 - err1^2| * 2^(2*fm1))
//   B (1 < E <= fm1 + 2) = RTN(|err2^2 - err1^2| * 2^(2*(fm1 + 1)))     RTN: half up
//   sum(B) - 4 * sum(A) < 0 -> fe1 (dynamic_exponent = 0), else fe2 (dynamic_exponent = 1)
// For comparison it also quantizes the row both ways with §5.1 (RTN) and reports the
// squared error of each. Those §5.1 steps differ from the §5.1.1 table in the clamp region.
//
// window.aaqDynamicSelect(values, W) -> { s, fm1, rows: [...], sumA, sumB, score, choice, sse1, sse2 }
// Widget markup: <div class="decoder" data-dynamic data-values="..." data-w="8"></div>
(function () {
  var buf = new DataView(new ArrayBuffer(4));
  function parts(x) {
    buf.setFloat32(0, x);
    var b = buf.getUint32(0);
    return { f32: buf.getFloat32(0), sign: b >>> 31, e: (b >>> 23) & 0xFF, sig: (b & 0x7FFFFF) | 0x800000 };
  }
  function rtn(x) { return Math.floor(x + 0.5); }

  // §5.1 quantization of one element to (fe, fm), returned as a value. Matches quantizer.js for W = 8.
  function quantize51(p, S, E, fe, fm) {
    var clampExp = (1 << fe) - 1, q, mag;
    if (E < clampExp) {
      q = rtn(p.sig / Math.pow(2, 23 - fm));                 // 1.M rounded to fm bits, in units of 2^-fm
      if (q === (1 << (fm + 1)) && E === 0) q = (1 << (fm + 1)) - 1; // edge 1 at E = 0: saturate
      mag = q / Math.pow(2, fm) * Math.pow(2, S - E);         // edge 1 at E > 0 gives the same value
    } else {
      var k = E - clampExp + 1;
      q = rtn(p.sig / Math.pow(2, 23 - fm + k));              // edge 2 gives the same value
      mag = q / Math.pow(2, fm) * Math.pow(2, S - clampExp + 1);
    }
    return p.sign ? -mag : mag;
  }

  function select(values, W) {
    var fm1 = W - 2; // signed: W - fe - 1 with fe = 1
    var ps = values.map(parts);
    var s = ps.reduce(function (m, p) { return Math.max(m, p.e); }, 0), S = s - 127;
    var sumA = 0, sumB = 0, sse1 = 0, sse2 = 0;
    var rows = ps.map(function (p) {
      var E = s - p.e, mag = Math.abs(p.f32);
      var step1 = Math.pow(2, S - Math.min(E, 1) - fm1);
      var step2 = Math.pow(2, S - Math.min(E, 3) - fm1 + 1);
      var err1 = mag - Math.floor(mag / step1) * step1;
      var err2 = mag - Math.floor(mag / step2) * step2;
      var diff = Math.abs(err2 * err2 - err1 * err1);
      var acc = E === 0 ? "A" : (E > 1 && E <= fm1 + 2 ? "B" : "none");
      var pre = acc === "A" ? diff * Math.pow(2, 2 * fm1) : (acc === "B" ? diff * Math.pow(2, 2 * (fm1 + 1)) : NaN);
      var term = isNaN(pre) ? null : rtn(pre);
      if (acc === "A") sumA += term;
      if (acc === "B") sumB += term;
      var q1 = quantize51(p, S, E, 1, fm1), q2 = quantize51(p, S, E, 2, fm1 - 1);
      sse1 += (q1 - p.f32) * (q1 - p.f32);
      sse2 += (q2 - p.f32) * (q2 - p.f32);
      return { x: p.f32, e: p.e, E: E, acc: acc, err1: err1, err2: err2, diff: diff, pre: pre, term: term, q1: q1, q2: q2 };
    });
    var score = sumB - 4 * sumA;
    return { s: s, fm1: fm1, rows: rows, sumA: sumA, sumB: sumB, score: score, choice: score < 0 ? 1 : 2, sse1: sse1, sse2: sse2 };
  }

  function num(v) { return String(+v.toPrecision(5)); }
  function sci(v) { return v === 0 ? "0" : v.toExponential(3); }
  function build(host) {
    var uid = "y" + Math.random().toString(36).slice(2, 7);
    host.innerHTML =
      '<div class="controls">' +
      '<label for="' + uid + 'v" style="flex:1 1 18rem">Row values (comma separated)<input id="' + uid + 'v" style="width:100%" value="' + (host.dataset.values || "") + '" autocomplete="off"></label>' +
      '<label for="' + uid + 'w">Element width W<select id="' + uid + 'w"><option value="8">8 (E1M6 or E2M5)</option><option value="6">6 (E1M4 or E2M3)</option></select></label>' +
      "</div>" +
      '<p class="result" aria-live="polite"></p>' +
      '<div class="table-wrap"><table><thead><tr><th class="num">Value</th><th class="num">E</th><th>Acc</th><th class="num">err1</th><th class="num">err2</th><th class="num">|err2² − err1²|</th><th class="num">Pre-RTN</th><th class="num">Term</th><th class="num">§5.1, fe1</th><th class="num">§5.1, fe2</th></tr></thead><tbody></tbody></table></div>';
    var wSel = host.querySelector("select");
    wSel.value = host.dataset.w || "8";
    var vIn = host.querySelector("#" + uid + "v"), out = host.querySelector(".result"), tbody = host.querySelector("tbody");
    function render() {
      var vals = vIn.value.split(/[,\s]+/).filter(Boolean).map(Number);
      if (!vals.length || vals.some(function (v) { return !isFinite(v); })) { out.textContent = "Enter finite numbers separated by commas."; tbody.innerHTML = ""; return; }
      if (vals.length > 128) vals = vals.slice(0, 128);
      var r = select(vals, +wSel.value);
      var better = r.sse1 < r.sse2 ? "fe1" : (r.sse2 < r.sse1 ? "fe2" : "neither");
      out.innerHTML = "s = " + r.s + " (unbiased " + (r.s - 127) + "), fm1 = " + r.fm1 +
        "<br>sum(A) = <strong>" + r.sumA + "</strong>, sum(B) = <strong>" + r.sumB + "</strong>, sum(B) − 4·sum(A) = <strong>" + r.score + "</strong>" +
        "<br>Choice: <strong>fe" + r.choice + "</strong>, dynamic_exponent = " + (r.choice - 1) +
        "<br>§5.1 squared error: fe1 " + sci(r.sse1) + ", fe2 " + sci(r.sse2) + " (lower: " + better + ")";
      tbody.innerHTML = r.rows.map(function (x) {
        var dc = x.acc === "none";
        return "<tr><td class=\"num\">" + num(x.x) + "</td><td class=\"num\">" + x.E + "</td><td>" + x.acc +
          "</td><td class=\"num\">" + (dc ? "–" : num(x.err1)) + "</td><td class=\"num\">" + (dc ? "–" : num(x.err2)) +
          "</td><td class=\"num\">" + (dc ? "–" : sci(x.diff)) + "</td><td class=\"num\">" + (dc ? "–" : num(x.pre)) +
          "</td><td class=\"num\">" + (dc ? "–" : x.term) + "</td><td class=\"num\">" + num(x.q1) + "</td><td class=\"num\">" + num(x.q2) + "</td></tr>";
      }).join("");
    }
    [vIn, wSel].forEach(function (el) { el.addEventListener("input", render); });
    render();
  }
  window.aaqDynamicSelect = select;
  function init() { document.querySelectorAll("[data-dynamic]").forEach(build); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
