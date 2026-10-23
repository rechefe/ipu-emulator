// Quiz component. Markup: <div class="panel" data-quiz="ID"></div> plus
// <script type="application/json" id="ID">[{prompt, options[], answer, why}]</script>.
// One try per question; feedback shows at once.
(function () {
  function build(host) {
    var data = JSON.parse(document.getElementById(host.dataset.quiz).textContent);
    var right = 0, done = 0;
    var score = document.createElement("p");
    score.className = "score";
    function updateScore() { score.textContent = "Score: " + right + " / " + data.length + (done < data.length ? "  (" + (data.length - done) + " left)" : "  (done)"); }
    data.forEach(function (q, qi) {
      var box = document.createElement("div"); box.className = "q";
      var p = document.createElement("p"); p.className = "q-prompt"; p.textContent = (qi + 1) + ". " + q.prompt;
      var opts = document.createElement("div"); opts.className = "q-options";
      var fb = document.createElement("p"); fb.className = "q-feedback"; fb.setAttribute("aria-live", "polite");
      q.options.forEach(function (text, oi) {
        var b = document.createElement("button"); b.type = "button"; b.textContent = text;
        b.addEventListener("click", function () {
          var buttons = opts.querySelectorAll("button");
          buttons.forEach(function (x) { x.disabled = true; });
          buttons[q.answer].classList.add("right");
          if (oi === q.answer) { right++; fb.textContent = "Correct. " + q.why; }
          else { b.classList.add("wrong"); fb.textContent = "Not quite. " + q.why; }
          done++; updateScore();
        });
        opts.appendChild(b);
      });
      box.appendChild(p); box.appendChild(opts); box.appendChild(fb); host.appendChild(box);
    });
    updateScore(); host.appendChild(score);
  }
  function revealers() {
    document.querySelectorAll("[data-reveal]").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var t = document.getElementById(btn.dataset.reveal); t.hidden = false; btn.disabled = true;
      });
    });
  }
  function init() { document.querySelectorAll("[data-quiz]").forEach(build); revealers(); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
