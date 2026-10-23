// LUT segment unpacker for the AaQ LUT load (stage-aaq-str.md §5.2, §6.3).
// One segment = payload[1023:0] = {r_acc[31], ..., r_acc[0]} (payload[32j +: 32] = r_acc[j]).
// Two readings of the same payload, because the spec gives two:
//   §5.2.1 flat: the table is one 4352-bit stream (256 entries x 17 bits); segment k is
//     stream bits [1024k + 1023 : 1024k]; entry i is stream bits [17i + 16 : 17i]. Segment 4
//     holds the last 256 table bits in payload[255:0] and the metadata in payload[330:256].
//   §6.3 pseudocode: for k <= 3, LUT[32k + j] = r_acc[j][16:0]; for k = 4, metadata = payload[74:0].
// Metadata layout (local bits): max_exp_num [7:0], neg_c [39:8], pos_c [71:40], asymetric [72], range_mode [74:73].
// Markup: <div class="decoder" data-lutpack data-seg="4"></div>
(function () {
  var ENTRY = 17, ENTRIES = 256, TABLE_BITS = ENTRY * ENTRIES; // 4352
  var M17 = (1n << 17n) - 1n;

  function payloadFromWords(words) { // words[j] = r_acc[j], 32 x uint32
    var p = 0n;
    for (var j = 31; j >= 0; j--) p = (p << 32n) | BigInt(words[j] >>> 0);
    return p;
  }
  function bits(p, lo, n) { return Number((p >> BigInt(lo)) & ((1n << BigInt(n)) - 1n)); }
  function f32(u) { var dv = new DataView(new ArrayBuffer(4)); dv.setUint32(0, u >>> 0); return dv.getFloat32(0); }
  function f32bits(x) { var dv = new DataView(new ArrayBuffer(4)); dv.setFloat32(0, x); return dv.getUint32(0); }

  function metadata(p, base) {
    return {
      max_exp_num: bits(p, base, 8),
      neg_c: f32(bits(p, base + 8, 32)),
      pos_c: f32(bits(p, base + 40, 32)),
      asymetric: bits(p, base + 72, 1),
      range_mode: bits(p, base + 73, 2)
    };
  }

  // §5.2.1: entries that segment k touches, with how many of their bits it carries.
  function flatReading(seg, p) {
    var segLo = 1024 * seg, segHi = Math.min(segLo + 1024, TABLE_BITS); // stream bits [segLo, segHi)
    var out = [];
    for (var i = Math.floor(segLo / ENTRY); i < ENTRIES && i * ENTRY < segHi; i++) {
      var eLo = i * ENTRY, eHi = eLo + ENTRY;
      var lo = Math.max(eLo, segLo), hi = Math.min(eHi, segHi);
      var part = bits(p, lo - segLo, hi - lo);
      out.push({ entry: i, have: hi - lo, from: lo - eLo, value: part, full: hi - lo === ENTRY });
    }
    return { entries: out, meta: seg === 4 ? metadata(p, 256) : null };
  }
  // §6.3: 32 entries per table segment from the low 17 bits of each word.
  function pseudoReading(seg, p) {
    if (seg === 4) return { entries: [], meta: metadata(p, 0) };
    var out = [];
    for (var j = 0; j < 32; j++) out.push({ entry: seg * 32 + j, have: 17, from: 0, value: bits(p, 32 * j, 17), full: true });
    return { entries: out, meta: null };
  }

  // Example payload for segment k under the flat reading: entry i = (i * 0x101 + 0x40) & 0x1FFFF,
  // and for k = 4 the metadata max_exp_num = 133, neg_c = -1.0, pos_c = 1.0, asymetric = 0, range_mode = 1.
  function exampleWords(seg) {
    var stream = 0n;
    for (var i = ENTRIES - 1; i >= 0; i--) stream = (stream << 17n) | (BigInt((i * 0x101 + 0x40) & 0x1FFFF) & M17);
    var p = (stream >> BigInt(1024 * seg)) & ((1n << 1024n) - 1n);
    if (seg === 4) {
      p &= (1n << 256n) - 1n;
      var meta = 133n | (BigInt(f32bits(-1.0)) << 8n) | (BigInt(f32bits(1.0)) << 40n) | (0n << 72n) | (1n << 73n);
      p |= meta << 256n;
    }
    var words = [];
    for (var j = 0; j < 32; j++) words.push(Number((p >> BigInt(32 * j)) & 0xFFFFFFFFn));
    return words;
  }

  function hex(v, n) { var s = (v >>> 0).toString(16).toUpperCase(); while (s.length < n) s = "0" + s; return s; }
  function metaRows(m) {
    if (!m) return "";
    return "max_exp_num = " + m.max_exp_num + ", neg_c = " + (+m.neg_c.toPrecision(7)) + ", pos_c = " + (+m.pos_c.toPrecision(7)) +
      ", asymetric = " + m.asymetric + ", range_mode = " + m.range_mode;
  }
  function entryTable(list) {
    if (!list.length) return "<p class=\"bitlegend\">No table entries in this reading.</p>";
    return '<div class="table-wrap" style="max-height:18rem;overflow:auto"><table><thead><tr><th class="num">Entry</th><th class="num">Bits carried</th><th class="num">Value</th></tr></thead><tbody>' +
      list.map(function (e) {
        return "<tr><td class=\"num\">" + e.entry + "</td><td class=\"num\">" + (e.full ? "17 of 17" : e.have + " of 17, from bit " + e.from) +
          "</td><td class=\"num\">0x" + hex(e.value, 5) + (e.full ? "" : " (partial)") + "</td></tr>";
      }).join("") + "</tbody></table></div>";
  }

  function build(host) {
    var uid = "l" + Math.random().toString(36).slice(2, 7);
    host.innerHTML =
      '<div class="controls">' +
      '<label for="' + uid + 's">lut_addr<select id="' + uid + 's"><option>0</option><option>1</option><option>2</option><option>3</option><option>4</option></select></label>' +
      '<label style="align-self:end"><button type="button" data-fill>Fill example for this segment</button></label>' +
      "</div>" +
      '<label for="' + uid + 'w" class="eyebrow">r_acc[0] … r_acc[31], 32 hex words</label>' +
      '<textarea id="' + uid + 'w" spellcheck="false" style="font-family:var(--font-mono);font-size:0.85rem;min-height:7rem"></textarea>' +
      '<p class="result" aria-live="polite"></p>' +
      '<div class="parties" style="grid-template-columns:repeat(auto-fit,minmax(16rem,1fr))">' +
      '<div class="party"><span class="who">§5.2.1 reading</span><h3>Flat 4352-bit stream</h3><div data-flat></div></div>' +
      '<div class="party"><span class="who">§6.3 reading</span><h3>Low 17 bits of each word</h3><div data-pseudo></div></div>' +
      "</div>";
    var sel = host.querySelector("select"), ta = host.querySelector("textarea"), out = host.querySelector(".result");
    var flatEl = host.querySelector("[data-flat]"), pseudoEl = host.querySelector("[data-pseudo]");
    sel.value = host.dataset.seg || "4";
    function fill() { ta.value = exampleWords(+sel.value).map(function (w) { return hex(w, 8); }).join(" "); render(); }
    function render() {
      var toks = ta.value.trim().split(/[\s,]+/).filter(Boolean);
      if (toks.length !== 32 || toks.some(function (t) { return !/^(0x)?[0-9a-f]{1,8}$/i.test(t); })) {
        out.textContent = "Enter exactly 32 hex words (up to 8 digits each). Found " + toks.length + ".";
        flatEl.innerHTML = pseudoEl.innerHTML = ""; return;
      }
      var words = toks.map(function (t) { return parseInt(t.replace(/^0x/i, ""), 16); });
      var seg = +sel.value, p = payloadFromWords(words);
      var a = flatReading(seg, p), b = pseudoReading(seg, p);
      out.textContent = "Segment " + seg + ": flat reading touches " + a.entries.length + " entries; §6.3 reading writes " + b.entries.length + ".";
      flatEl.innerHTML = entryTable(a.entries) + (a.meta ? '<p class="mono">payload[330:256]: ' + metaRows(a.meta) + "</p>" : "");
      pseudoEl.innerHTML = entryTable(b.entries) + (b.meta ? '<p class="mono">payload[74:0]: ' + metaRows(b.meta) + "</p>" : "");
    }
    host.querySelector("[data-fill]").addEventListener("click", fill);
    sel.addEventListener("input", fill);
    ta.addEventListener("input", render);
    fill();
  }
  window.aaqLut = { payloadFromWords: payloadFromWords, flatReading: flatReading, pseudoReading: pseudoReading, exampleWords: exampleWords };
  function init() { document.querySelectorAll("[data-lutpack]").forEach(build); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
