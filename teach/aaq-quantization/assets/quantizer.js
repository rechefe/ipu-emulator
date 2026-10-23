// AaQ row quantizer: MAS-stage-aaq-str.md §5.1, signed, W = 8, static fe.
// Integer arithmetic on the FP32 bits, so the result is exact.
// RTN ties round half up, as MAS §5.1 and ARCH AaQ §4.3.2.2 now state.
//
// window.aaqQuantizeRow(values, fe) -> { s, elements: [{ x, f32, e, E, branch, edge, exp, man, byte, decoded, err }] }
// Widget markup: <div class="decoder" data-quantizer data-values="3.2, 0.4, -1.99" data-fe="4"></div>
(function () {
  var buf = new DataView(new ArrayBuffer(4));
  function f32bits(x) { buf.setFloat32(0, x); return buf.getUint32(0); }
  function f32(x) { buf.setFloat32(0, x); return buf.getFloat32(0); }
  function rtn(x) { return Math.floor(x + 0.5); } // half up; x >= 0

  function quantizeRow(values, fe) {
    var fm = 7 - fe, clampExp = (1 << fe) - 1;
    var parts = values.map(function (x) {
      var b = f32bits(x);
      return { x: x, f32: f32(x), sign: b >>> 31, e: (b >>> 23) & 0xFF, sig: (b & 0x7FFFFF) | 0x800000 };
    });
    // §5.1: s = max raw exponent over the row (the spec's "0..254" is read as the 128 elements).
    var s = parts.reduce(function (m, p) { return Math.max(m, p.e); }, 0);
    var elements = parts.map(function (p) {
      var E = s - p.e, exp, man, branch, edge = "";
      if (E < clampExp) {
        branch = "direct";
        exp = E;
        man = rtn(p.sig / Math.pow(2, 23 - fm)) - (1 << fm); // RTN(1.M) to fm bits, fraction part
        if (man === (1 << fm)) {
          if (E > 0) { exp = E - 1; man = 0; edge = "edge 1: carry, one binade up"; }
          else { exp = 0; man = (1 << fm) - 1; edge = "edge 1 at E = 0: saturate"; }
        }
      } else {
        branch = "clamp";
        exp = clampExp;
        var k = E - clampExp + 1;
        man = rtn(p.sig / Math.pow(2, 23 - fm + k)); // RTN(1.M >> k) to fm bits
        if (man === (1 << fm)) { exp = clampExp - 1; man = 0; edge = "edge 2: carry into the smallest direct exponent"; }
      }
      var byte = (p.sign << 7) | (exp << fm) | man;
      var decoded = window.aaqDecode ? window.aaqDecode(byte, fe, s).value : NaN;
      return { x: p.x, f32: p.f32, e: p.e, E: E, branch: branch, edge: edge, exp: exp, man: man, byte: byte, decoded: decoded, err: decoded - p.f32 };
    });
    return { s: s, elements: elements };
  }

  function hex2(v) { return "0x" + (v < 16 ? "0" : "") + v.toString(16).toUpperCase(); }
  function num(v) { return String(+v.toPrecision(6)); }
  function build(host) {
    var uid = "q" + Math.random().toString(36).slice(2, 7);
    host.innerHTML =
      '<div class="controls">' +
      '<label for="' + uid + 'v" style="flex:1 1 18rem">Row values (comma separated)<input id="' + uid + 'v" style="width:100%" value="' + (host.dataset.values || "3.2, 0.4") + '" autocomplete="off"></label>' +
      '<label for="' + uid + 'f">Table fe<select id="' + uid + 'f"></select></label>' +
      "</div>" +
      '<p class="result" aria-live="polite"></p>' +
      '<div class="table-wrap"><table><thead><tr><th class="num">Value</th><th class="num">e</th><th class="num">E = s − e</th><th>Branch</th><th class="num">Exp</th><th class="num">M</th><th class="num">Byte</th><th class="num">Decodes to</th><th>Note</th></tr></thead><tbody></tbody></table></div>';
    var sel = host.querySelector("select");
    for (var f = 1; f <= 6; f++) { var o = document.createElement("option"); o.value = f; o.textContent = f + "  (E" + f + "M" + (7 - f) + ")"; sel.appendChild(o); }
    sel.value = host.dataset.fe || "4";
    var vIn = host.querySelector("#" + uid + "v"), out = host.querySelector(".result"), tbody = host.querySelector("tbody");
    function render() {
      var vals = vIn.value.split(/[,\s]+/).filter(Boolean).map(Number);
      if (!vals.length || vals.some(function (v) { return !isFinite(v); })) { out.textContent = "Enter finite numbers separated by commas."; tbody.innerHTML = ""; return; }
      if (vals.length > 128) vals = vals.slice(0, 128);
      var r = quantizeRow(vals, +sel.value);
      out.innerHTML = "Row scale s = <strong>" + r.s + "</strong> (top binade [" + num(Math.pow(2, r.s - 127)) + ", " + num(Math.pow(2, r.s - 126)) + "))";
      tbody.innerHTML = r.elements.map(function (el) {
        return "<tr><td class=\"num\">" + num(el.f32) + "</td><td class=\"num\">" + el.e + "</td><td class=\"num\">" + el.E + "</td><td>" + el.branch +
          "</td><td class=\"num\">" + el.exp + "</td><td class=\"num\">" + el.man + "</td><td class=\"num\">" + hex2(el.byte) + "</td><td class=\"num\">" + num(el.decoded) + "</td><td>" + el.edge + "</td></tr>";
      }).join("");
    }
    [vIn, sel].forEach(function (el) { el.addEventListener("input", render); });
    render();
  }
  window.aaqQuantizeRow = quantizeRow;
  function init() { document.querySelectorAll("[data-quantizer]").forEach(build); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
