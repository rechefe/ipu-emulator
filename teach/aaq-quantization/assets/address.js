// Row-address splitter. Layout is the OQ-8 proposal (not yet decided):
//   row_address = CR[base] + LR[offset] = { table_id[23:20], page[19:10], row[9:0] }
//   tag = { ctx, table_id, page }, bank position = page % 16 (cache unit §9).
// Markup: <div class="decoder" data-address data-cr="200000" data-lr="1029" data-ctx="0"></div>
(function () {
  function split(cr, lr) {
    var addr = (cr + lr) & 0xFFFFFF;
    return { addr: addr, table: (addr >> 20) & 0xF, page: (addr >> 10) & 0x3FF, row: addr & 0x3FF };
  }
  function hex(v, n) { var s = v.toString(16).toUpperCase(); while (s.length < n) s = "0" + s; return "0x" + s; }
  function build(host) {
    var uid = "a" + Math.random().toString(36).slice(2, 7);
    host.innerHTML =
      '<div class="controls">' +
      '<label for="' + uid + 'c">CR[base] (hex)<input id="' + uid + 'c" value="' + (host.dataset.cr || "200000") + '" maxlength="6" autocomplete="off"></label>' +
      '<label for="' + uid + 'l">LR[offset] (decimal)<input id="' + uid + 'l" type="number" min="0" value="' + (host.dataset.lr || "1029") + '"></label>' +
      '<label for="' + uid + 'x">active_ctx<select id="' + uid + 'x"><option value="0">0</option><option value="1">1</option></select></label>' +
      "</div>" +
      '<div class="bits addr" aria-label="address fields"></div>' +
      '<p class="bitlegend">Violet: table_id [23:20]. Amber: page [19:10]. Blue: row [9:0].</p>' +
      '<p class="result" aria-live="polite"></p>';
    var cIn = host.querySelector("#" + uid + "c"), lIn = host.querySelector("#" + uid + "l"), xIn = host.querySelector("#" + uid + "x");
    xIn.value = host.dataset.ctx || "0";
    var bitsEl = host.querySelector(".bits"), out = host.querySelector(".result");
    function render() {
      var cr = parseInt(cIn.value, 16), lr = parseInt(lIn.value, 10), ctx = +xIn.value;
      if (isNaN(cr) || isNaN(lr) || cr < 0 || lr < 0) { bitsEl.innerHTML = ""; out.textContent = "Enter a hex CR value and a decimal LR value."; return; }
      var r = split(cr, lr), html = "";
      for (var i = 23; i >= 0; i--) {
        var cls = i >= 20 ? "e" : (i >= 10 ? "p" : "m");
        html += '<span class="' + cls + '">' + ((r.addr >> i) & 1) + "</span>";
      }
      bitsEl.innerHTML = html;
      out.innerHTML = "row_address = " + hex(r.addr, 6) +
        "<br>table_id = <strong>" + r.table + "</strong>, page = <strong>" + r.page + "</strong>, row = <strong>" + r.row + "</strong>" +
        "<br>tag = {ctx " + ctx + ", table " + r.table + ", page " + r.page + "}, bank position = page % 16 = " + (r.page % 16);
    }
    [cIn, lIn, xIn].forEach(function (el) { el.addEventListener("input", render); });
    render();
  }
  window.aaqSplitAddress = split;
  function init() { document.querySelectorAll("[data-address]").forEach(build); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
