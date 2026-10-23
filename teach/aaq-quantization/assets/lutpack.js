// LUT segment unpacker for LOAD_LUT (ARCH-stage-aaq-str.md §4.2).
// One segment = payload[1023:0] = {input_data[31], ..., input_data[0]} (payload[32j +: 32] = R_ACC[j]).
// The table is one flat 4352-bit stream (256 entries x 17 bits). Segment k (0-3) is stream bits
// [1024k + 1023 : 1024k]; entry i is stream bits [17i + 16 : 17i]. Segment 4 holds the last 256
// table bits in payload[255:0] and the 46-bit LUT metadata in payload[301:256] (§4.2.2):
//   max_exp_num [7:0], neg_c [24:8] (17 bits), pos_c [41:25] (17 bits), range_mode [43:42],
//   neg_mode [44], pos_mode [45].
// Entry points (§4.2.3): step = 2^max_exp_num / 256. Version 1: i*step. Version 2: -i*step.
// Version 3: 128 negative entries at interval 2*step ending at -0, then 128 positive from +0.
// The spec does not say which entry index holds which point in version 3; this assumes the
// listed order (entries 0-127 negative, ascending; 128-255 positive).
// Markup: <div class="decoder" data-lutpack data-seg="4"></div>
(function () {
  var ENTRY = 17, ENTRIES = 256, TABLE_BITS = ENTRY * ENTRIES; // 4352
  var M17 = (1n << 17n) - 1n;

  function payloadFromWords(words) { // words[j] = R_ACC[j], 32 x uint32
    var p = 0n;
    for (var j = 31; j >= 0; j--) p = (p << 32n) | BigInt(words[j] >>> 0);
    return p;
  }
  function bits(p, lo, n) { return Number((p >> BigInt(lo)) & ((1n << BigInt(n)) - 1n)); }

  function metadata(p) { // ARCH §4.2.2, local bits from payload bit 256
    var b = 256;
    return {
      max_exp_num: bits(p, b, 8),
      neg_c: bits(p, b + 8, 17),
      pos_c: bits(p, b + 25, 17),
      range_mode: bits(p, b + 42, 2),
      neg_mode: bits(p, b + 44, 1),
      pos_mode: bits(p, b + 45, 1)
    };
  }

  // Entries that segment k touches, with how many of their bits it carries.
  function segmentEntries(seg, p) {
    var segLo = 1024 * seg, segHi = Math.min(segLo + 1024, TABLE_BITS); // stream bits [segLo, segHi)
    var out = [];
    for (var i = Math.floor(segLo / ENTRY); i < ENTRIES && i * ENTRY < segHi; i++) {
      var eLo = i * ENTRY, eHi = eLo + ENTRY;
      var lo = Math.max(eLo, segLo), hi = Math.min(eHi, segHi);
      out.push({ entry: i, have: hi - lo, from: lo - eLo, value: bits(p, lo - segLo, hi - lo), full: hi - lo === ENTRY });
    }
    return { entries: out, meta: seg === 4 ? metadata(p) : null };
  }

  // §4.2.3: the x value an entry stands for, or null for range_mode 3 (not defined).
  function entryPoint(i, rangeMode, maxExp) {
    var step = Math.pow(2, maxExp) / 256;
    if (rangeMode === 0) return i * step;
    if (rangeMode === 1) return i === 0 ? 0 : -i * step;
    if (rangeMode === 2) return i < 128 ? (i === 127 ? -0 : -(127 - i) * 2 * step) : (i - 128) * 2 * step;
    return null;
  }

  // Example: entry i = (i * 0x101 + 0x40) & 0x1FFFF; segment 4 metadata max_exp_num = 2,
  // neg_c = 0x00000, pos_c = 0x10000, range_mode = 2 (version 3), neg_mode = 1, pos_mode = 1.
  function exampleWords(seg) {
    var stream = 0n;
    for (var i = ENTRIES - 1; i >= 0; i--) stream = (stream << 17n) | (BigInt((i * 0x101 + 0x40) & 0x1FFFF) & M17);
    var p = (stream >> BigInt(1024 * seg)) & ((1n << 1024n) - 1n);
    if (seg === 4) {
      p &= (1n << 256n) - 1n;
      var meta = 2n | (0x00000n << 8n) | (0x10000n << 25n) | (2n << 42n) | (1n << 44n) | (1n << 45n);
      p |= meta << 256n;
    }
    var words = [];
    for (var j = 0; j < 32; j++) words.push(Number((p >> BigInt(32 * j)) & 0xFFFFFFFFn));
    return words;
  }

  function hex(v, n) { var s = (v >>> 0).toString(16).toUpperCase(); while (s.length < n) s = "0" + s; return s; }
  function num(v) { return v === null ? "not defined" : (Object.is(v, -0) ? "−0" : String(+v.toPrecision(6))); }
  var VERSION = ["version 1: positive x", "version 2: negative x", "version 3: both sides", "not defined"];

  function build(host) {
    var uid = "l" + Math.random().toString(36).slice(2, 7);
    host.innerHTML =
      '<div class="controls">' +
      '<label for="' + uid + 's">lut_addr<select id="' + uid + 's"><option>0</option><option>1</option><option>2</option><option>3</option><option>4</option></select></label>' +
      '<label for="' + uid + 'r">range_mode for entry points<select id="' + uid + 'r"><option value="0">0</option><option value="1">1</option><option value="2">2</option></select></label>' +
      '<label for="' + uid + 'm">max_exp_num<input id="' + uid + 'm" type="number" min="0" max="20" value="2"></label>' +
      '<label style="align-self:end"><button type="button" data-fill>Fill example for this segment</button></label>' +
      "</div>" +
      '<label for="' + uid + 'w" class="eyebrow">R_ACC[0] … R_ACC[31], 32 hex words</label>' +
      '<textarea id="' + uid + 'w" spellcheck="false" style="font-family:var(--font-mono);font-size:0.85rem;min-height:7rem"></textarea>' +
      '<p class="result" aria-live="polite"></p>' +
      '<p class="mono" data-meta></p>' +
      '<div class="table-wrap" style="max-height:18rem;overflow:auto"><table><thead><tr><th class="num">Entry</th><th class="num">Bits carried</th><th class="num">Value</th><th class="num">Stands for x =</th></tr></thead><tbody></tbody></table></div>';
    var sel = host.querySelector("#" + uid + "s"), rm = host.querySelector("#" + uid + "r"), mx = host.querySelector("#" + uid + "m");
    var ta = host.querySelector("textarea"), out = host.querySelector(".result"), metaEl = host.querySelector("[data-meta]"), tbody = host.querySelector("tbody");
    sel.value = host.dataset.seg || "4";
    rm.value = "2";
    function fill() { ta.value = exampleWords(+sel.value).map(function (w) { return hex(w, 8); }).join(" "); render(); }
    function render() {
      var toks = ta.value.trim().split(/[\s,]+/).filter(Boolean);
      if (toks.length !== 32 || toks.some(function (t) { return !/^(0x)?[0-9a-f]{1,8}$/i.test(t); })) {
        out.textContent = "Enter exactly 32 hex words (up to 8 digits each). Found " + toks.length + ".";
        metaEl.textContent = ""; tbody.innerHTML = ""; return;
      }
      var words = toks.map(function (t) { return parseInt(t.replace(/^0x/i, ""), 16); });
      var seg = +sel.value, r = segmentEntries(seg, payloadFromWords(words));
      if (r.meta) { rm.value = String(Math.min(r.meta.range_mode, 2)); mx.value = String(r.meta.max_exp_num); }
      var rangeMode = +rm.value, maxExp = +mx.value;
      out.textContent = "Segment " + seg + " carries " + r.entries.length + " entries, " +
        r.entries.filter(function (e) { return !e.full; }).length + " of them split across a segment boundary.";
      metaEl.textContent = r.meta
        ? "payload[301:256]: max_exp_num = " + r.meta.max_exp_num + ", neg_c = 0x" + hex(r.meta.neg_c, 5) + ", pos_c = 0x" + hex(r.meta.pos_c, 5) +
          ", range_mode = " + r.meta.range_mode + " (" + VERSION[r.meta.range_mode] + "), neg_mode = " + r.meta.neg_mode + ", pos_mode = " + r.meta.pos_mode
        : "No LUT metadata in this segment.";
      tbody.innerHTML = r.entries.map(function (e) {
        return "<tr><td class=\"num\">" + e.entry + "</td><td class=\"num\">" + (e.full ? "17 of 17" : e.have + " of 17, from bit " + e.from) +
          "</td><td class=\"num\">0x" + hex(e.value, 5) + (e.full ? "" : " (partial)") + "</td><td class=\"num\">" + num(entryPoint(e.entry, rangeMode, maxExp)) + "</td></tr>";
      }).join("");
    }
    host.querySelector("[data-fill]").addEventListener("click", fill);
    sel.addEventListener("input", fill);
    [rm, mx].forEach(function (el) { el.addEventListener("input", render); });
    ta.addEventListener("input", render);
    fill();
  }
  window.aaqLut = { payloadFromWords: payloadFromWords, segmentEntries: segmentEntries, entryPoint: entryPoint, exampleWords: exampleWords };
  function init() { document.querySelectorAll("[data-lutpack]").forEach(build); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
