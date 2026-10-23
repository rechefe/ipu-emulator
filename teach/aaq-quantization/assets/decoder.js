// AaQ byte decoder: signed, W = 8. The decode inverts stage-aaq-str.md §5.1;
// the spec itself has no decode section.
//   Exp < 2^fe - 1 : value = ±(1 + M/2^fm) · 2^(s - 127 - Exp)
//   Exp = 2^fe - 1 : value = ±(M/2^fm)     · 2^(s - 127 - 2^fe + 2)
// Markup: <div class="decoder" data-decoder data-byte="1D" data-fe="4" data-scale="128"></div>
(function () {
  function decode(byte, fe, s) {
    var fm = 7 - fe;
    var sign = (byte >> 7) & 1;
    var exp = (byte >> fm) & ((1 << fe) - 1);
    var man = byte & ((1 << fm) - 1);
    var clampExp = (1 << fe) - 1;
    var mag, formula;
    if (exp < clampExp) {
      mag = (1 + man / (1 << fm)) * Math.pow(2, s - 127 - exp);
      formula = "(1 + " + man + "/" + (1 << fm) + ") × 2^(" + s + " − 127 − " + exp + ")";
    } else {
      var p = s - 127 - (1 << fe) + 2;
      mag = (man / (1 << fm)) * Math.pow(2, p);
      formula = "(" + man + "/" + (1 << fm) + ") × 2^(" + p + ")   clamp branch";
    }
    return { sign: sign, exp: exp, man: man, fm: fm, value: sign ? -mag : mag, formula: (sign ? "−" : "+") + formula };
  }
  function build(host) {
    var uid = "d" + Math.random().toString(36).slice(2, 7);
    host.innerHTML =
      '<div class="controls">' +
      '<label for="' + uid + 'b">Byte (hex)<input id="' + uid + 'b" value="' + (host.dataset.byte || "1D") + '" maxlength="2" autocomplete="off"></label>' +
      '<label for="' + uid + 'f">Table format: exponent bits <span class="mono">fe</span><select id="' + uid + 'f"></select></label>' +
      '<label for="' + uid + 's">Row scale <span class="mono">s</span> (e8m0)<input id="' + uid + 's" type="number" min="0" max="254" value="' + (host.dataset.scale || "128") + '"></label>' +
      "</div>" +
      '<div class="bits" aria-label="bit fields"></div>' +
      '<p class="bitlegend">Grey: sign. Violet: exponent field (width from the table). Blue: mantissa (the rest).</p>' +
      '<p class="result" aria-live="polite"></p>';
    var sel = host.querySelector("select");
    for (var f = 1; f <= 6; f++) {
      var o = document.createElement("option"); o.value = f; o.textContent = f + "  (E" + f + "M" + (7 - f) + ")"; sel.appendChild(o);
    }
    sel.value = host.dataset.fe || "4";
    var byteIn = host.querySelector("#" + uid + "b"), sIn = host.querySelector("#" + uid + "s");
    var bitsEl = host.querySelector(".bits"), out = host.querySelector(".result");
    function render() {
      var b = parseInt(byteIn.value, 16), fe = +sel.value, s = +sIn.value;
      if (isNaN(b) || b < 0 || b > 255 || isNaN(s) || s < 0 || s > 254) {
        bitsEl.innerHTML = ""; out.textContent = "Enter a byte 00–FF and a scale 0–254."; return;
      }
      var r = decode(b, fe, s), html = "";
      for (var i = 7; i >= 0; i--) {
        var cls = i === 7 ? "s" : (i >= r.fm ? "e" : "m");
        html += '<span class="' + cls + '">' + ((b >> i) & 1) + "</span>";
      }
      bitsEl.innerHTML = html;
      out.innerHTML = "Exp = " + r.exp + ", M = " + r.man + "<br>" + r.formula + "<br>value = <strong>" + r.value.toPrecision(6).replace(/\.?0+$/, "") + "</strong>";
    }
    [byteIn, sel, sIn].forEach(function (el) { el.addEventListener("input", render); });
    render();
  }
  window.aaqDecode = decode;
  function init() { document.querySelectorAll("[data-decoder]").forEach(build); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
