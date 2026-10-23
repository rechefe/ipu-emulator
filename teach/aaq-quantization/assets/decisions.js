// Decision board. Markup: <div class="panel" data-decisions="ID"></div> plus
// <script type="application/json" id="ID">[{id, title, question, options: [{label, detail}], recommended}]</script>.
// Each pick updates a plain-text summary the viewer copies back into the chat.
// Picks persist per viewer in localStorage when it is available (wrapped in try/catch).
(function () {
  function load(key) { try { return JSON.parse(localStorage.getItem(key) || "{}"); } catch (e) { return {}; } }
  function save(key, v) { try { localStorage.setItem(key, JSON.stringify(v)); } catch (e) { /* storage blocked: picks last for this visit only */ } }

  function build(host) {
    var id = host.dataset.decisions;
    var items = JSON.parse(document.getElementById(id).textContent);
    var key = "aaq-decisions-" + id;
    var picks = load(key);
    var summary = document.createElement("pre");
    summary.className = "result";
    summary.style.whiteSpace = "pre-wrap";
    summary.tabIndex = 0;
    var copy = document.createElement("button");
    copy.type = "button";
    copy.textContent = "Copy summary";
    var status = document.createElement("p");
    status.className = "q-feedback";
    status.setAttribute("aria-live", "polite");

    function render() {
      var lines = ["My answers to the open AaQ design questions:"];
      items.forEach(function (it) {
        var p = picks[it.id];
        lines.push(it.id + " " + it.title + ": " + (p === undefined ? "(no answer yet)" : it.options[p].label + (p === it.recommended ? " (recommended)" : "")));
      });
      summary.textContent = lines.join("\n");
    }

    items.forEach(function (it) {
      var box = document.createElement("div"); box.className = "q";
      var h = document.createElement("p"); h.className = "q-prompt"; h.textContent = it.id + " · " + it.title;
      var q = document.createElement("p"); q.textContent = it.question;
      var opts = document.createElement("div"); opts.className = "q-options";
      var detail = document.createElement("p"); detail.className = "q-feedback";
      var buttons = it.options.map(function (o, i) {
        var b = document.createElement("button"); b.type = "button";
        b.textContent = o.label + (i === it.recommended ? "  · recommended" : "");
        b.setAttribute("aria-pressed", "false");
        b.addEventListener("click", function () { picks[it.id] = i; save(key, picks); mark(); render(); });
        opts.appendChild(b);
        return b;
      });
      function mark() {
        var p = picks[it.id];
        buttons.forEach(function (b, i) {
          var on = i === p;
          b.classList.toggle("right", on);
          b.setAttribute("aria-pressed", on ? "true" : "false");
        });
        detail.textContent = p === undefined ? "Pick one. " + it.options[it.recommended].detail : it.options[p].detail;
      }
      mark();
      box.appendChild(h); box.appendChild(q); box.appendChild(opts); box.appendChild(detail);
      host.appendChild(box);
    });

    copy.addEventListener("click", function () {
      var text = summary.textContent;
      function fallback() {
        var r = document.createRange(); r.selectNodeContents(summary);
        var sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(r);
        status.textContent = "Copy is blocked here. The summary is selected: press Ctrl+C (or Cmd+C).";
      }
      try {
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text).then(function () { status.textContent = "Copied. Paste it into the chat."; }, fallback);
        } else { fallback(); }
      } catch (e) { fallback(); }
    });

    var h3 = document.createElement("h3"); h3.textContent = "Your summary";
    host.appendChild(h3); host.appendChild(summary); host.appendChild(copy); host.appendChild(status);
    render();
  }
  function init() { document.querySelectorAll("[data-decisions]").forEach(build); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
