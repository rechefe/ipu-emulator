// Side-by-side reading of one byte: the emulator's Ex (biased exponent,
// src/ipu-isa/src/ipu_isa/formats.py _fp8_decode) and AaQ's scale-relative
// ExM(7-x) (decoder.js, inverted from stage-aaq-str.md §5.1). Needs decoder.js first.
// Markup: <div class="decoder" data-compare data-byte="1D" data-x="4" data-scale="128"></div>
(function () {
  function emuDecode(code, x) {
    var m = 7 - x, sign = (code >> 7) & 1;
    var e = (code >> m) & ((1 << x) - 1), man = code & ((1 << m) - 1);
    if (e === (1 << x) - 1) return { value: NaN, how: "exponent all ones → NaN" };
    var bias = (1 << (x - 1)) - 1, v, how;
    if (e === 0) { v = (man / (1 << m)) * Math.pow(2, 1 - bias); how = "exponent 0 → subnormal: (" + man + "/" + (1 << m) + ") × 2^(1 − " + bias + ")"; }
    else { v = (1 + man / (1 << m)) * Math.pow(2, e - bias); how = "(1 + " + man + "/" + (1 << m) + ") × 2^(" + e + " − " + bias + ")"; }
    return { value: sign ? -v : v, how: how };
  }
  function fmt(v) { return isNaN(v) ? "NaN" : String(+v.toPrecision(6)); }
  function build(host) {
    var uid = "c" + Math.random().toString(36).slice(2, 7);
    host.innerHTML =
      '<div class="controls">' +
      '<label for="' + uid + 'b">Byte (hex)<input id="' + uid + 'b" value="' + (host.dataset.byte || "1D") + '" maxlength="2" autocomplete="off"></label>' +
      '<label for="' + uid + 'x">Exponent bits x<select id="' + uid + 'x"></select></label>' +
      '<label for="' + uid + 's">AaQ row scale s<input id="' + uid + 's" type="number" min="0" max="254" value="' + (host.dataset.scale || "128") + '"></label>' +
      "</div>" +
      '<div class="bits" aria-label="bit fields"></div>' +
      '<div class="table-wrap"><table><thead><tr><th>Reading</th><th>Value</th><th>How</th></tr></thead><tbody></tbody></table></div>';
    var sel = host.querySelector("select");
    for (var x = 1; x <= 6; x++) { var o = document.createElement("option"); o.value = x; o.textContent = "E" + x + "M" + (7 - x); sel.appendChild(o); }
    sel.value = host.dataset.x || "4";
    var bIn = host.querySelector("#" + uid + "b"), sIn = host.querySelector("#" + uid + "s");
    var bitsEl = host.querySelector(".bits"), tbody = host.querySelector("tbody");
    function render() {
      var b = parseInt(bIn.value, 16), x = +sel.value, s = +sIn.value;
      if (isNaN(b) || b < 0 || b > 255 || isNaN(s) || s < 0 || s > 254) { bitsEl.innerHTML = ""; tbody.innerHTML = '<tr><td colspan="3">Enter a byte 00–FF and a scale 0–254.</td></tr>'; return; }
      var html = "", m = 7 - x;
      for (var i = 7; i >= 0; i--) html += '<span class="' + (i === 7 ? "s" : (i >= m ? "e" : "m")) + '">' + ((b >> i) & 1) + "</span>";
      bitsEl.innerHTML = html;
      var emu = emuDecode(b, x), aaq = window.aaqDecode(b, x, s);
      tbody.innerHTML =
        '<tr><td>Emulator E' + x + '</td><td class="num"><strong>' + fmt(emu.value) + '</strong></td><td class="mono">' + emu.how + "</td></tr>" +
        '<tr><td>AaQ E' + x + "M" + (7 - x) + ", s = " + s + '</td><td class="num"><strong>' + fmt(aaq.value) + '</strong></td><td class="mono">' + aaq.formula + "</td></tr>";
    }
    [bIn, sel, sIn].forEach(function (el) { el.addEventListener("input", render); });
    render();
  }
  window.emuDecode = emuDecode;
  function init() { document.querySelectorAll("[data-compare]").forEach(build); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
