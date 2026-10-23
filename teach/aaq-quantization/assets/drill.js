// Answer-checking drill. Markup: <div class="panel" data-drill="ID"></div> plus
// <script type="application/json" id="ID">[{prompt, answer, solution}]</script>.
// Answers compare as hex bytes when they parse as hex ("0x04", "04", "4"), else as trimmed text.
(function () {
  function norm(s) {
    s = String(s).trim().toLowerCase().replace(/^0x/, "");
    return /^[0-9a-f]{1,2}$/.test(s) ? parseInt(s, 16) : s;
  }
  function build(host) {
    var tasks = JSON.parse(document.getElementById(host.dataset.drill).textContent);
    tasks.forEach(function (t, i) {
      var uid = host.dataset.drill + "-" + i;
      var box = document.createElement("div"); box.className = "q";
      box.innerHTML =
        '<label class="q-prompt" for="' + uid + '">' + (i + 1) + ". " + t.prompt + "</label>" +
        '<div class="controls" style="display:flex;gap:0.5rem;flex-wrap:wrap">' +
        '<input id="' + uid + '" class="mono" autocomplete="off" placeholder="0x__" style="width:7rem;padding:0.4rem 0.5rem;border:1px solid var(--rule);border-radius:6px;background:var(--bg);color:var(--ink);font-family:var(--font-mono)">' +
        '<button type="button">Check</button><button type="button" data-show>Show solution</button></div>' +
        '<p class="q-feedback" aria-live="polite"></p>';
      var input = box.querySelector("input"), fb = box.querySelector(".q-feedback");
      var buttons = box.querySelectorAll("button");
      function check() {
        if (!input.value.trim()) { fb.textContent = "Type a byte first, for example 0x2A."; return; }
        if (norm(input.value) === norm(t.answer)) { input.style.borderColor = "var(--good)"; fb.textContent = "Correct. " + t.solution; }
        else { input.style.borderColor = "var(--bad)"; fb.textContent = "Not yet. Try again, or show the solution."; }
      }
      buttons[0].addEventListener("click", check);
      input.addEventListener("keydown", function (e) { if (e.key === "Enter") check(); });
      buttons[1].addEventListener("click", function () { fb.textContent = "Answer " + t.answer + ". " + t.solution; });
      host.appendChild(box);
    });
  }
  function init() { document.querySelectorAll("[data-drill]").forEach(build); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
