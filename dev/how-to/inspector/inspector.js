(() => {
"use strict";
/* The Cycle Inspector (E08-17): replays traces the docs build records
   (docs/inspector, E08-16). Its data, fetched relative to this page:
   data/kernels.json (the curated list), data/<id>.json (one trace, the
   document ipu_gen.inspector_trace writes) and data/isa.json (the instruction
   reference the editor extension also reads). */
const VIEWER_SCHEMA = 2;  // the trace document version this viewer reads
const KERNELS_SCHEMA = 1;  // the kernel list version this viewer reads
let KERNELS = [];
// Slot order and instance counts come from isa.json (slotOrder, slotCount).
let SLOTS = [];
let SLOT_NAME = {};
let LANES = [];
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const hex4 = (n) => "0x" + n.toString(16).padStart(4, "0");
const sv = (k) => `var(--s-${k})`;

let ISA = null, ISA_BY_NAME = {};
let T = null;            // current trace
let states = [];         // states[k] = register map after k cycles
let pos = 0;             // cycles executed
let timer = null;
let breakpoints = new Set();
let hitCounts = [];
let viewModes = {};
const cache = {};
const store = { get(k) { try { return localStorage.getItem(k); } catch { return null; } }, set(k, v) { try { localStorage.setItem(k, v); } catch {} } };

function hexToBytes(h) { const a = new Uint8Array(h.length / 2); for (let i = 0; i < a.length; i++) a[i] = parseInt(h.substr(i * 2, 2), 16); return a; }

/* ---------- data ---------- */
async function getJSON(path) {
  if (cache[path]) return cache[path];
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return (cache[path] = await r.json());
}

function buildStates(t) {
  const init = {};
  for (const k in t.init) init[k] = hexToBytes(t.init[k]);
  const out = [init];
  let prev = init;
  for (const c of t.cycles) {
    const s = Object.assign({}, prev);
    for (const reg in c.d) {
      const a = prev[reg].slice();
      if (reg === "lr" || reg === "cr") {
        const dv = new DataView(a.buffer);
        for (const i in c.d[reg]) dv.setUint32(+i * 4, c.d[reg][i] >>> 0, true);
      } else {
        for (const [off, h] of c.d[reg]) a.set(hexToBytes(h), off);
      }
      s[reg] = a;
    }
    out.push(s);
    prev = s;
  }
  return out;
}

function laneMap(entry) {
  const m = {};
  for (const s of entry.slots) m[s.slot + s.instance] = s;
  return m;
}

// isa.json (ipu-gen vscode-data): every mnemonic's forms. The viewer lists
// each slot's real instructions in opcode order; a pseudo-instruction opens
// as one.
function loadISA(data) {
  SLOTS = data.slotOrder;
  SLOT_NAME = Object.fromEntries(SLOTS.map((s) => [s, s.toUpperCase()]));
  LANES = SLOTS.flatMap((s) => Array.from({ length: data.slotCount[s] }, (_, i) => [s, i]));
  ISA = Object.fromEntries(SLOTS.map((s) => [s, []]));
  for (const [name, forms] of Object.entries(data.instructions)) {
    for (const form of forms) {
      if (form.pseudo) continue;
      const ins = { name, opcode: form.opcode, operands: form.operands, doc: form.doc, slot: form.slot };
      ISA[form.slot].push(ins);
      if (name !== "NOP") ISA_BY_NAME[name.toUpperCase()] = ins;
    }
  }
  for (const s of SLOTS) ISA[s].sort((a, b) => a.opcode - b.opcode);
  TL_ROWS = ["macs", ...SLOTS, "xmem"];
  $("lanes").style.setProperty("--lane-count", LANES.length);
  readColors();
}

/* ---------- kernel load ---------- */
async function loadKernel(id) {
  stop();
  const shown = T ? T.id : null;
  const select = (k) => { for (const b of document.querySelectorAll(".kbtn")) b.setAttribute("aria-selected", b.dataset.id === k); };
  $("listing").innerHTML = '<li class="loading">Loading trace…</li>';
  select(id);
  try {
    const [isa, t] = await Promise.all([getJSON("data/isa.json"), getJSON(`data/${id}.json`)]);
    if (!ISA) loadISA(isa);
    if (t.schemaVersion !== VIEWER_SCHEMA) {
      throw new Error(t.schemaVersion > VIEWER_SCHEMA
        ? `this trace is newer than the viewer (trace version ${t.schemaVersion}, viewer ${VIEWER_SCHEMA}); reload the page`
        : `this trace is older than the viewer (trace version ${t.schemaVersion}, viewer ${VIEWER_SCHEMA}); rebuild the docs`);
    }
    T = t; T.id = id;
    // NX8 (the hardware mode): 128-byte rows, each with a metadata word; FP32: the emulator-only mode.
    T.wideMode = t.meta.emulator_only ? "f32" : "native";
    T.lut = t.meta.lut;  // NX8: per dynamic_exponent, each code's value at unit scale
    states = buildStates(t);
    hitCounts = new Array(t.program.length).fill(0);
    for (const c of t.cycles) hitCounts[c.pc]++;
    breakpoints = new Set();
    viewModes = {};
    T.lanes = t.program.map(laneMap);
    T.used = new Set(t.program.flatMap((p) => p.slots.map((s) => s.op)));
    pos = 0;
    store.set("ipu-kernel", id);
    renderKernelInfo();
    renderListing();
    renderSource();
    renderISA();
    buildVecPanels();
    drawTimelineBase();
    update();
  } catch (e) {
    // The previous kernel stays loaded: keep its panels and button, but say why.
    select(shown);
    if (T) renderListing();
    $("listing").insertAdjacentHTML("afterbegin", `<li class="err">Could not load the trace for ${esc(id)}: ${esc(e.message)}</li>`);
  }
}

function renderKernelInfo() {
  const k = KERNELS.find((x) => x.id === T.id);
  const dt = `NX8 ${T.meta.variant}`;
  $("kinfo").innerHTML = `<strong>${esc(k.title)} · ${esc(k.label)}</strong>
    <span class="pill">${esc(T.source_path)} · case ${esc(T.case)}</span>
    <span class="pill">${T.program.length} instructions</span>
    <span class="pill">${T.total_cycles.toLocaleString()} cycles</span>
    <span class="pill" title="Multiplier lanes used over the whole run, out of 128 per cycle">MAC utilization ${Math.round(100 * T.cycles.reduce((a, c) => a + (c.L || 0), 0) / (128 * T.cycles.length))}%</span>
    ${T.wideMode === "native" ? `<span class="pill">format ${esc(dt)}</span>` : ""}
    <span class="pill">${T.wideMode === "native" ? "hardware mode: 128-byte rows, each with a metadata word" : "emulator-only FP32 mode, 32-bit lanes"}</span>
    <span class="desc">${esc(k.description)}</span>`;
  const tr = $("trunc");
  tr.hidden = !T.truncated;
  if (T.truncated) tr.textContent = `Showing the first ${T.cycles.length.toLocaleString()} of ${T.total_cycles.toLocaleString()} cycles (the docs build keeps the first ${T.cycles.length.toLocaleString()}).`;
  $("lst-meta").textContent = `${T.program.length} words · ${T.layout.reduce((a, f) => a + f[1], 0)} bits each`;
}

/* ---------- listing ---------- */
function chipHTML(s) {
  return `<span class="chip" style="--sc:${sv(s.slot)}" title="${esc(SLOT_NAME[s.slot] || s.slot)} slot"><b>${esc(s.op)}</b>${s.args.length ? `<i>${esc(s.args.join(" "))}</i>` : ""}</span>`;
}
function renderListing() {
  const maxHits = Math.max(1, ...hitCounts);
  const html = [];
  T.program.forEach((p, i) => {
    if (p.label) html.push(`<li class="lbl">${esc(p.label)}:</li>`);
    const w = Math.round((hitCounts[i] / maxHits) * 40);
    html.push(`<li class="row" data-pc="${i}" id="row-${i}" title="Click to run until this instruction executes">
      <button class="bp" data-bp="${i}" aria-label="Toggle breakpoint at ${hex4(i)}" title="Toggle breakpoint"></button>
      <span class="addr">${i.toString(16).padStart(4, "0")}</span>
      <span class="chips">${p.slots.map(chipHTML).join("")}</span>
      <span class="hits" title="Executed ${hitCounts[i]} times in this trace">×${hitCounts[i]}<span class="bar" style="width:${w}px"></span></span></li>`);
  });
  $("listing").innerHTML = html.join("");
}
$("listing").addEventListener("click", (e) => {
  const bp = e.target.closest(".bp");
  if (bp) {
    const i = +bp.dataset.bp;
    breakpoints.has(i) ? breakpoints.delete(i) : breakpoints.add(i);
    bp.classList.toggle("on", breakpoints.has(i));
    drawTimelineBase(); drawPlayhead();
    return;
  }
  const row = e.target.closest(".row");
  if (!row || !T) return;
  const pc = +row.dataset.pc;
  stop();
  for (let k = pos + 1; k < T.cycles.length; k++) if (T.cycles[k].pc === pc) { seek(k); return; }
  for (let k = 0; k <= pos; k++) if (T.cycles[k] && T.cycles[k].pc === pc) { seek(k); return; }
});

/* ---------- source + ISA ---------- */
function renderSource() {
  $("src-title").textContent = T.source_path.split("/").pop();
  $("src").innerHTML = esc(T.source)
    .replace(/(\{#[\s\S]*?#\})/g, '<span class="c">$1</span>')
    .replace(/^(\s*#.*)$/gm, '<span class="c">$1</span>');
}
function renderISA() {
  const html = [];
  for (const slot of SLOTS) {
    const list = ISA[slot] || [];
    html.push(`<section class="panel isa-slot" style="--sc:${sv(slot)}"><header><h2>${SLOT_NAME[slot]} slot</h2><span class="eyebrow">${list.length} opcodes</span></header><ul class="isa-list">`);
    for (const ins of list) {
      const used = T && T.used.has(ins.name.toUpperCase());
      html.push(`<li><button data-op="${esc(ins.name)}"><span class="nm">${esc(ins.name)}</span><span class="oc">${used ? '<span class="used">in this kernel</span> · ' : ""}op ${ins.opcode}</span><span class="sm">${esc(ins.doc ? ins.doc.title : "")}</span></button></li>`);
    }
    html.push("</ul></section>");
  }
  $("isa").innerHTML = html.join("");
}
$("isa").addEventListener("click", (e) => { const b = e.target.closest("[data-op]"); if (b) openDoc(b.dataset.op); });

/* ---------- drawer ---------- */
function openDoc(op, inst) {
  const ins = ISA_BY_NAME[op.toUpperCase()];
  const d = $("drawer");
  if (!ins) {
    d.innerHTML = `<header style="--sc:var(--muted)"><div><h2>${esc(op)}</h2><div class="dt">Pseudo-instruction, expanded by the assembler. See PSEUDO_INSTRUCTION_SPEC.</div></div><button class="xbtn" id="dclose">Close</button></header>`;
  } else {
    const doc = ins.doc || {};
    const sec = (t, body) => body ? `<div><div class="eyebrow">${t}</div>${body}</div>` : "";
    d.innerHTML = `<header style="--sc:${sv(ins.slot)}"><div><h2>${esc(ins.name)}</h2><div class="dt">${esc(doc.title || "")} · ${esc(SLOT_NAME[ins.slot])} slot · opcode ${ins.opcode}</div></div><button class="xbtn" id="dclose">Close</button></header>
      <div class="body">
        ${inst ? sec("As executed here", `<pre>${esc(inst.op + (inst.args.length ? " " + inst.args.join(" ") : ""))}</pre>`) : ""}
        ${doc.summary ? `<p style="margin:0">${esc(doc.summary)}</p>` : ""}
        ${sec("Syntax", doc.syntax ? `<pre>${esc(doc.syntax)}</pre>` : "")}
        ${sec("Operands", ins.operands.length ? `<ul>${(doc.operands && doc.operands.length ? doc.operands : ins.operands.map((o) => o.name)).map((o) => `<li>${esc(o)}</li>`).join("")}</ul><div class="mono" style="font-size:11px;color:var(--muted);margin-top:4px">${ins.operands.map((o) => `${esc(o.name)}: ${esc(o.type)}`).join(" · ")}</div>` : "<div>None</div>")}
        ${sec("Operation", doc.operation ? `<pre>${esc(doc.operation)}</pre>` : "")}
        ${sec("Example", doc.example ? `<pre>${esc(doc.example)}</pre>` : "")}
        ${sec("Notes", doc.notes ? `<p style="margin:0;white-space:pre-wrap">${esc(doc.notes)}</p>` : "")}
        <div class="hint">From <span class="mono">instruction_spec.py</span>.</div>
      </div>`;
  }
  d.hidden = false; $("scrim").hidden = false;
  $("dclose").addEventListener("click", closeDoc);
  $("dclose").focus();
}
function closeDoc() { $("drawer").hidden = true; $("scrim").hidden = true; }
$("scrim").addEventListener("click", closeDoc);

/* ---------- VLIW panel ---------- */
function fieldGroup(name) { const m = name.match(/^([a-z_]+?_inst(?:_\d)?)_token/); return m ? m[1] : name; }
function shortField(name) { return name.replace(/^[a-z_]+?_inst(?:_\d)?_token_\d+_/, "").replace(/_(field|type|token)$/, "").replace(/_inst_opcode$/, " opcode"); }

function renderVLIW(pcExec) {
  const lanesEl = $("lanes"), bitsEl = $("bits");
  if (pcExec == null) {
    $("vliw-title").textContent = "Instruction word";
    $("vliw-sub").textContent = "Nothing executed yet · press Step";
    lanesEl.innerHTML = LANES.map(([s, i]) => `<div class="lane" style="--sc:${sv(s)}"><span class="ln">${SLOT_NAME[s]}${s === "lr" ? i : ""}</span><span class="op">—</span></div>`).join("");
    bitsEl.innerHTML = ""; $("word").textContent = "";
    return;
  }
  const entry = T.program[pcExec], lm = T.lanes[pcExec];
  const active = entry.slots.length;
  $("vliw-title").textContent = `Cycle ${pos} executed PC ${hex4(pcExec)}`;
  $("vliw-sub").textContent = `${active} of ${LANES.length} slots issued${entry.label ? " · label " + entry.label : ""}`;
  lanesEl.innerHTML = LANES.map(([s, i]) => {
    const ins = lm[s + i];
    const name = `${SLOT_NAME[s]}${s === "lr" ? i : ""}`;
    return ins
      ? `<button class="lane active" style="--sc:${sv(s)}" data-lane="${s + i}" title="Open ${esc(ins.op)} reference"><span class="ln">${name}</span><span class="op">${esc(ins.op)}</span><span class="args">${esc(ins.args.join(" "))}</span></button>`
      : `<div class="lane" style="--sc:${sv(s)}"><span class="ln">${name}</span><span class="op">NOP</span></div>`;
  }).join("");
  // bit strip
  let word = BigInt("0x" + entry.word);
  const total = T.layout.reduce((a, f) => a + f[1], 0);
  let shift = total;
  const groups = [];
  for (const [name, w, slot] of T.layout) {
    shift -= w;
    const val = Number((word >> BigInt(shift)) & ((1n << BigInt(w)) - 1n));
    const g = fieldGroup(name);
    if (!groups.length || groups[groups.length - 1].g !== g) groups.push({ g, slot, w: 0, fields: [] });
    const grp = groups[groups.length - 1];
    grp.w += w;
    grp.fields.push({ name, w, val, hi: shift + w - 1, lo: shift });
  }
  bitsEl.innerHTML = groups.map((g) => `<div class="bgroup" style="--w:${g.w};--sc:${sv(g.slot)}"><span class="gl">${esc(g.g.replace("_inst", "").replace(/_(\d)$/, "$1"))}</span><div class="bfields">${
    g.fields.map((f) => `<div class="bf${f.val ? " nz" : ""}" style="--w:${f.w}" data-tip="${esc(`${shortField(f.name)}  [${f.hi}:${f.lo}]  ${f.w} bit${f.w > 1 ? "s" : ""}\nvalue ${f.val}  (0b${f.val.toString(2).padStart(f.w, "0")})`)}">${f.w >= 3 ? f.val : ""}</div>`).join("")
  }</div></div>`).join("");
  $("word").textContent = "0x" + entry.word.padStart(Math.ceil(total / 4), "0");
}
$("lanes").addEventListener("click", (e) => {
  const b = e.target.closest("[data-lane]"); if (!b || !pos) return;
  const ins = T.lanes[T.cycles[pos - 1].pc][b.dataset.lane];
  if (ins) openDoc(ins.op, ins);
});

/* ---------- registers ---------- */
function toI32(u) { return u | 0; }
function renderScalars() {
  const s = states[pos], p = pos ? states[pos - 1] : null;
  for (const reg of ["lr", "cr"]) {
    const dv = new DataView(s[reg].buffer), pv = p ? new DataView(p[reg].buffer) : null;
    let html = "";
    for (let i = 0; i < 16; i++) {
      const v = dv.getUint32(i * 4, true);
      const chg = pv && pv.getUint32(i * 4, true) !== v;
      const ro = reg === "cr" && i < 2;
      html += `<div class="sreg${chg ? " chg" : ""}${ro ? " ro" : ""}" title="${reg.toUpperCase()}${i} = ${toI32(v)} (0x${v.toString(16).padStart(8, "0")})${chg ? `, was ${toI32(pv.getUint32(i * 4, true))}` : ""}"><span class="n">${reg.toUpperCase()}${i}</span><span class="x">${toI32(v)}</span></div>`;
    }
    $("g-" + reg).innerHTML = html;
  }
}

function vecDefs() {
  const wide = T.wideMode !== "native";
  const q = T.meta.nx8;
  const dn = q && `NX8 ${T.meta.variant} (${q.dynamic ? "dynamic, fe 1 or 2 per row" : `fe ${q.exp_format}`}, ${q.num_bits} bits${q.sign ? ", signed" : ""})`;
  // An NX8 register's metadata: `meta` names the register holding one 16-bit word per 128 elements.
  return wide ? [
    { key: "R0", reg: "r_wide_debug", off: 0, len: 512, mode: "f32", cols: 16, note: "128 × FP32 lanes (emulator-only mode)" },
    { key: "R1", reg: "r_wide_debug", off: 512, len: 512, mode: "f32", cols: 16, note: "128 × FP32 lanes (emulator-only mode)" },
    { key: "R_CYCLIC", reg: "r_cyclic_wide_debug", off: 0, len: 2048, mode: "f32", cols: 32, note: "512 × FP32 lanes in four 128-element slots (emulator-only mode)" },
    { key: "MULT_RES", reg: "mult_res", off: 0, len: 512, mode: "f32", cols: 16, note: "multiply stage result" },
    { key: "R_ACC", reg: "r_acc", off: 0, len: 512, mode: "f32", cols: 16, note: "accumulator" },
    { key: "R_MASK", reg: "r_mask", off: 0, len: 128, mode: "bits", cols: 64, note: "8 slots × 128 mask bits" },
  ] : [
    { key: "R0", reg: "r", off: 0, len: 128, mode: "nx8", meta: { reg: "r_meta", off: 0 }, cols: 16, note: `128 × ${dn}, scaled by R0's metadata` },
    { key: "R1", reg: "r", off: 128, len: 128, mode: "nx8", meta: { reg: "r_meta", off: 2 }, cols: 16, note: `128 × ${dn}, scaled by R1's metadata` },
    { key: "R_CYCLIC", reg: "r_cyclic", off: 0, len: 512, mode: "nx8", meta: { reg: "r_cyclic_meta", off: 0 }, cols: 32, note: `512 × ${dn}, four 128-element slots, each with its row's metadata` },
    { key: "MULT_RES", reg: "mult_res", off: 0, len: 512, mode: "f32", cols: 16, note: "128 × FP32 products" },
    { key: "R_ACC", reg: "r_acc", off: 0, len: 512, mode: "f32", cols: 16, note: "128 × FP32 accumulator; ACTIVATE.QUANTIZE stores it as one NX8 row" },
    { key: "R_MASK", reg: "r_mask", off: 0, len: 128, mode: "bits", cols: 64, note: "8 slots × 128 mask bits" },
  ];
}
/** The metadata words of a vector panel in state `s`: one per 128 elements, or null. */
function metaWords(v, s) {
  if (!v.meta || !s[v.meta.reg]) return null;
  const b = s[v.meta.reg], n = Math.ceil(v.len / 128), out = new Array(n);
  for (let k = 0; k < n; k++) out[k] = b[v.meta.off + 2 * k] | (b[v.meta.off + 2 * k + 1] << 8);
  return out;
}
/** A metadata word as the page shows it: scale[7:0] (e8m0, bias 127), dynamic_exponent[8]. */
function metaText(w) { return `scale 2^${(w & 0xff) - 127}${T.meta.nx8 && T.meta.nx8.dynamic ? `, dynamic_exponent ${(w >> 8) & 1}` : ""}`; }
let VDEFS = [];
function buildVecPanels() {
  VDEFS = vecDefs();
  $("vecs").innerHTML = VDEFS.map((v) => `<div class="vec" data-key="${v.key}">
    <div class="vec-h"><b>${v.key}</b><span class="meta" id="vm-${v.key}"></span>
      <span class="vmode" role="group" aria-label="${v.key} element view">${[...(T.lut && v.meta ? ["nx8"] : []), "i8", "i32", "f32", "bits"].map((m) => `<button data-v="${v.key}" data-m="${m}" aria-pressed="false">${m}</button>`).join("")}</span></div>
    <canvas id="cv-${v.key}" data-key="${v.key}"></canvas>
    <div class="range"><span id="lo-${v.key}"></span><span class="legend-scale"></span><span id="hi-${v.key}"></span></div>
    <div class="meta hint">${esc(v.note)}</div></div>`).join("");
}
$("vecs").addEventListener("click", (e) => {
  const b = e.target.closest("[data-m]"); if (!b) return;
  viewModes[b.dataset.v] = b.dataset.m; renderVectors();
});

function elems(bytes, mode, meta) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (mode === "i8") return Array.from(bytes, (b) => (b << 24) >> 24);
  if (mode === "nx8") {  // the code's value at unit scale, times 2^(scale - 127) of its row
    return Array.from(bytes, (b, i) => {
      const w = meta ? meta[i >> 7] : 127;
      const table = T.lut[Math.min((w >> 8) & 1, T.lut.length - 1)];
      return table[b] * Math.pow(2, (w & 0xff) - 127);
    });
  }
  if (mode === "i32") { const n = bytes.length >> 2, a = new Array(n); for (let i = 0; i < n; i++) a[i] = dv.getInt32(i * 4, true); return a; }
  if (mode === "f32") { const n = bytes.length >> 2, a = new Array(n); for (let i = 0; i < n; i++) a[i] = dv.getFloat32(i * 4, true); return a; }
  const a = new Array(bytes.length * 8); for (let i = 0; i < bytes.length; i++) for (let b = 0; b < 8; b++) a[i * 8 + b] = (bytes[i] >> b) & 1; return a;
}
const COLORS = {};
function readColors() {
  const cs = getComputedStyle(document.documentElement);
  for (const k of ["pos", "neg", "cell", "ink", "cur-line", "panel", "accent", "line", "muted", "bg"]) COLORS[k] = cs.getPropertyValue("--" + k).trim();
  for (const s of SLOTS) COLORS["s-" + s] = cs.getPropertyValue("--s-" + s).trim();
}
function hexRGB(h) {
  const c = document.createElement("canvas").getContext("2d"); c.fillStyle = h; const v = c.fillStyle;
  if (v.startsWith("#")) return [parseInt(v.slice(1, 3), 16), parseInt(v.slice(3, 5), 16), parseInt(v.slice(5, 7), 16)];
  const m = v.match(/\d+/g); return m ? m.slice(0, 3).map(Number) : [128, 128, 128];
}
function mix(a, b, t) { return `rgb(${Math.round(a[0] + (b[0] - a[0]) * t)},${Math.round(a[1] + (b[1] - a[1]) * t)},${Math.round(a[2] + (b[2] - a[2]) * t)})`; }
function fmt(v, mode) {
  if (mode === "f32" || mode === "nx8") { if (!isFinite(v)) return String(v); const a = Math.abs(v); return a !== 0 && (a < 1e-3 || a >= 1e5) ? v.toExponential(2) : +v.toFixed(4) + ""; }
  return String(v);
}
const vcache = {};
function renderVectors() {
  const s = states[pos], p = pos ? states[pos - 1] : null;
  const cPos = hexRGB(COLORS.pos), cNeg = hexRGB(COLORS.neg), cCell = hexRGB(COLORS.cell);
  for (const v of VDEFS) {
    const mode = viewModes[v.key] || v.mode;
    for (const b of document.querySelectorAll(`[data-v="${v.key}"]`)) b.setAttribute("aria-pressed", b.dataset.m === mode);
    const bytes = s[v.reg].subarray(v.off, v.off + v.len);
    const pbytes = p ? p[v.reg].subarray(v.off, v.off + v.len) : null;
    const meta = metaWords(v, s), pmeta = p ? metaWords(v, p) : null;
    const vals = elems(bytes, mode, meta);
    const pvals = pbytes ? elems(pbytes, mode, pmeta) : null;
    const n = vals.length;
    const cols = mode === "bits" ? 64 : n >= 512 ? 32 : 16;
    const rows = Math.ceil(n / cols);
    let maxAbs = 0, lo = Infinity, hi = -Infinity, changed = 0;
    for (let i = 0; i < n; i++) { const x = vals[i]; if (isFinite(x)) { maxAbs = Math.max(maxAbs, Math.abs(x)); lo = Math.min(lo, x); hi = Math.max(hi, x); } if (pvals && !Object.is(pvals[i], vals[i])) changed++; }
    const cv = $("cv-" + v.key);
    const cell = mode === "bits" ? 6 : cols >= 32 ? 10 : 18;
    const W = cols * cell, H = rows * cell;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    if (cv.width !== W * dpr || cv.height !== H * dpr) { cv.width = W * dpr; cv.height = H * dpr; }
    cv.style.aspectRatio = `${W} / ${H}`;
    const g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = COLORS.panel; g.fillRect(0, 0, W, H);
    const gap = mode === "bits" ? 0.5 : 1;
    for (let i = 0; i < n; i++) {
      const x = (i % cols) * cell, y = Math.floor(i / cols) * cell;
      const val = vals[i];
      let col;
      if (mode === "bits") col = val ? COLORS.accent : COLORS.cell;
      else if (!isFinite(val)) col = COLORS.ink;
      else if (maxAbs === 0 || val === 0) col = COLORS.cell;
      else { const t = Math.pow(Math.abs(val) / maxAbs, 0.6); col = mix(cCell, val > 0 ? cPos : cNeg, t); }
      g.fillStyle = col; g.fillRect(x + gap / 2, y + gap / 2, cell - gap, cell - gap);
    }
    if (pvals && changed && changed < n) {
      g.strokeStyle = COLORS.ink; g.lineWidth = 1.2;
      for (let i = 0; i < n; i++) if (!Object.is(pvals[i], vals[i])) {
        const x = (i % cols) * cell, y = Math.floor(i / cols) * cell;
        g.strokeRect(x + 1, y + 1, cell - 2, cell - 2);
      }
    }
    if (v.key === "R_CYCLIC") {  // four 128-element slots
      g.strokeStyle = COLORS.ink; g.lineWidth = 1.5; g.font = `600 ${Math.max(8, cell - 2)}px ${getComputedStyle(document.body).fontFamily}`;
      const per = n / 4 / cols;
      for (let q = 0; q < 4; q++) {
        const y = q * per * cell;
        if (q) { g.beginPath(); g.moveTo(0, y); g.lineTo(W, y); g.stroke(); }
      }
    }
    vcache[v.key] = { vals, pvals, cols, cell, mode, W, H };
    const metaChg = meta && pmeta && meta.some((w, k) => w !== pmeta[k]);
    const metaHTML = meta ? ` · <span class="${metaChg ? "chgd" : ""}" title="The metadata word of each 128-element row">${meta.map(metaText).join(" | ")}</span>` : "";
    $("vm-" + v.key).innerHTML = (changed ? `<span class="chgd">${changed === n ? "all" : changed} ${mode === "bits" ? "bits" : "elements"} changed</span>` : `${n} ${mode === "bits" ? "bits" : "elements"}`) + metaHTML;
    $("lo-" + v.key).textContent = mode === "bits" ? "0" : isFinite(lo) ? fmt(maxAbs ? -maxAbs : 0, mode) : "";
    $("hi-" + v.key).textContent = mode === "bits" ? "1" : isFinite(hi) ? fmt(maxAbs, mode) : "";
  }
}

/* tooltips */
const tip = $("tip");
function showTip(text, x, y) { tip.textContent = text; tip.hidden = false; const r = tip.getBoundingClientRect(); tip.style.left = Math.min(x + 14, innerWidth - r.width - 8) + "px"; tip.style.top = Math.max(8, y - r.height - 10) + "px"; }
document.addEventListener("mousemove", (e) => {
  const c = e.target.closest && e.target.closest("canvas[data-key]");
  if (c) {
    const d = vcache[c.dataset.key]; if (!d) return;
    const r = c.getBoundingClientRect();
    const cx = Math.floor(((e.clientX - r.left) / r.width) * d.W / d.cell), cy = Math.floor(((e.clientY - r.top) / r.height) * d.H / d.cell);
    const i = cy * d.cols + cx;
    if (i < 0 || i >= d.vals.length) { tip.hidden = true; return; }
    const was = d.pvals && !Object.is(d.pvals[i], d.vals[i]) ? `\nwas ${fmt(d.pvals[i], d.mode)}` : "";
    showTip(`${c.dataset.key}[${i}] = ${fmt(d.vals[i], d.mode)}  (${d.mode})${was}`, e.clientX, e.clientY);
    return;
  }
  const t = e.target.closest && e.target.closest("[data-tip]");
  if (t) { showTip(t.dataset.tip, e.clientX, e.clientY); return; }
  if (!tip.hidden) tip.hidden = true;
});

/* ---------- effects ---------- */
const REG_LABEL = { r: "R0/R1", r_wide_debug: "R0/R1 (32-bit)", r_cyclic: "R_CYCLIC", r_cyclic_wide_debug: "R_CYCLIC (32-bit)", r_meta: "R0/R1 metadata", r_cyclic_meta: "R_CYCLIC metadata", r_mask: "R_MASK", r_acc: "R_ACC", mult_res: "MULT_RES", aaq_lut: "AaQ LUT" };
function renderEffects() {
  if (!pos) { $("eff-regs").innerHTML = '<span class="none">No cycle executed yet.</span>'; $("eff-mem").innerHTML = '<span class="none">—</span>'; return; }
  const c = T.cycles[pos - 1], s = states[pos], p = states[pos - 1];
  const out = [];
  for (const reg in c.d) {
    if (reg === "lr" || reg === "cr") {
      const dv = new DataView(s[reg].buffer), pv = new DataView(p[reg].buffer);
      for (const i in c.d[reg]) out.push(`${reg.toUpperCase()}${i}: ${toI32(pv.getUint32(i * 4, true))} → <b>${toI32(dv.getUint32(i * 4, true))}</b>`);
    } else {
      const bytes = c.d[reg].reduce((a, sp) => a + sp[1].length / 2, 0);
      out.push(`${REG_LABEL[reg] || reg}: ${bytes} bytes`);
    }
  }
  const next = T.cycles[pos];
  const branch = next && next.pc !== c.pc + 1 ? `<span style="color:var(--s-cond)">branch taken → ${hex4(next.pc)}</span>` : "";
  if (c.L) out.unshift(`<span style="color:var(--s-mult)">MULT: ${c.L} / 128 lanes multiplied</span>`);
  $("eff-regs").innerHTML = (out.length ? out.map((x) => `<span>${x}</span>`).join("") : '<span class="none">No register changed.</span>') + branch;
  const mem = c.m.map(([k, a, n]) => `<span class="${k === "R" ? "memR" : "memW"}">${k === "R" ? "read " : "write"} ${n} B @ 0x${a.toString(16).padStart(6, "0")} <span style="color:var(--muted)">(row ${Math.floor(a / 128)})</span></span>`)
    .concat((c.M || []).map(([a, w]) => `<span class="memW">metadata of row ${Math.floor(a / 128)}: ${esc(metaText(w))}</span>`));
  $("eff-mem").innerHTML = mem.length ? mem.join("") : '<span class="none">No XMEM access this cycle.</span>';
}

/* ---------- hardware pipeline ---------- */
let pview = "datapath";
for (const b of document.querySelectorAll("[data-pv]")) b.addEventListener("click", () => {
  pview = b.dataset.pv;
  for (const x of document.querySelectorAll("[data-pv]")) x.setAttribute("aria-pressed", x === b);
  $("pv-datapath").hidden = pview !== "datapath"; $("pv-timing").hidden = pview !== "timing";
  renderPipeline();
});

function changedChips(c) {
  const out = new Set();
  if (!c) return out;
  for (const reg in c.d) {
    if (reg === "lr") out.add("LR");
    else if (reg === "cr") out.add("CR");
    else if (reg === "r" || reg === "r_wide_debug") {
      const half = reg === "r" ? 128 : 512;
      for (const [off, h] of c.d[reg]) { if (off < half) out.add("R0"); if (off + h.length / 2 > half) out.add("R1"); }
    }
    else if (reg === "r_meta") { for (const [off, h] of c.d[reg]) { if (off < 2) out.add("R0"); if (off + h.length / 2 > 2) out.add("R1"); } }
    else if (reg.startsWith("r_cyclic")) out.add("R_CYCLIC");
    else if (reg === "r_mask") out.add("R_MASK");
    else if (reg === "mult_res") out.add("MULT_RES");
    else if (reg === "r_acc") out.add("R_ACC");
    else if (reg === "aaq_lut") out.add("LUT");
  }
  return out;
}
const DP_BLOCKS = [
  { k: "ctrl", x: 10, w: 190, t: "CTRL", s: "fetch · branch · LR ALUs", col: "lr", lanes: ["cond0", "lr0", "lr1", "lr2", "break0"], chips: ["PC", "LR", "CR"] },
  { k: "agent", x: 222, w: 122, t: "Cache agent", s: "hit / miss · XMEM read", col: "load", lanes: ["load0"], chips: [] },
  { k: "mult", x: 366, w: 168, t: "MULT", s: "128-lane multiply", col: "mult", lanes: ["mult0"], chips: ["R0", "R1", "R_CYCLIC", "R_MASK", "MULT_RES"] },
  { k: "acc", x: 556, w: 128, t: "ACC", s: "accumulate", col: "acc", lanes: ["acc0"], chips: ["R_ACC"] },
  // The AaQ stage stores its own row (ADR 0007); ACTIVATE.EXACT is its emulator-only twin.
  { k: "aaq", x: 706, w: 284, t: "AAQ", s: "activate · quantize · store row → XMEM", col: "aaq", lanes: ["aaq0", "aaq_exact0"], chips: ["LUT"] },
];
const BY = 34, BH = 178, XY = 262, XH = 46;
function clip(s, n) { return s.length > n ? s.slice(0, n - 1) + "…" : s; }
function arrow(x1, y1, x2, y2) {
  const a = Math.atan2(y2 - y1, x2 - x1), L = 8, W = 4.5;
  const bx = x2 - L * Math.cos(a), by = y2 - L * Math.sin(a);
  return `<line x1="${x1}" y1="${y1}" x2="${bx}" y2="${by}"/><polygon points="${x2},${y2} ${bx - W * Math.sin(a)},${by + W * Math.cos(a)} ${bx + W * Math.sin(a)},${by - W * Math.cos(a)}"/>`;
}
function edge(on, col, body, label, lx, ly, anchor = "middle", extra = "") {
  return `<g class="edge${on ? " on" : ""}${extra}" style="--sc:${sv(col)}">${body}${label ? `<text x="${lx}" y="${ly}" text-anchor="${anchor}">${esc(label)}</text>` : ""}</g>`;
}
function fmtBytes(n) { return n >= 1024 ? (n / 1024) + " KiB" : n + " B"; }
function renderDatapath() {
  const c = pos ? T.cycles[pos - 1] : null;
  const lm = c ? T.lanes[c.pc] : {};
  const chg = changedChips(c);
  if (c) { chg.add("PC"); }
  const next = T.cycles[pos];
  const taken = c && next && next.pc !== c.pc + 1;
  const reads = c ? c.m.filter((m) => m[0] === "R") : [], writes = c ? c.m.filter((m) => m[0] === "W") : [];
  const on = (b) => b.lanes.some((l) => lm[l]);
  const parts = [];
  // pipeline flow arrows between blocks
  const flowY = BY + 22;
  for (let i = 0; i < DP_BLOCKS.length - 1; i++) {
    const a = DP_BLOCKS[i], b = DP_BLOCKS[i + 1];
    parts.push(edge(on(b), b.col, arrow(a.x + a.w, flowY, b.x, flowY), "", 0, 0));
  }
  // branch loop over CTRL
  const ctrl = DP_BLOCKS[0];
  parts.push(edge(taken, "cond", `<path d="M ${ctrl.x + ctrl.w - 30} ${BY} C ${ctrl.x + ctrl.w - 30} ${BY - 26}, ${ctrl.x + 40} ${BY - 26}, ${ctrl.x + 40} ${BY - 4}"/><polygon points="${ctrl.x + 40},${BY} ${ctrl.x + 35.5},${BY - 9} ${ctrl.x + 44.5},${BY - 9}"/>`,
    taken ? `branch taken → ${hex4(next.pc)}` : "PC + 1", ctrl.x + ctrl.w / 2, BY - 22));
  // blocks
  for (const b of DP_BLOCKS) {
    const act = on(b);
    const g = [`<g class="${act ? "on" : ""}" style="--sc:${sv(b.col)}"><rect class="box" x="${b.x}" y="${BY}" width="${b.w}" height="${BH}" rx="6"/>`,
      `<text class="bt" x="${b.x + 10}" y="${BY + 20}">${b.t}</text><text class="bs" x="${b.x + 10}" y="${BY + 36}">${b.s}</text>`];
    const ops = b.lanes.map((l) => lm[l] && { l, ins: lm[l] }).filter(Boolean);
    const maxc = Math.floor((b.w - 18) / 6.6);
    let y = BY + 56;
    if (!ops.length) g.push(`<text class="idle" x="${b.x + 10}" y="${y}">${c ? "idle" : "—"}</text>`);
    const lineH = b.k === "ctrl" && ops.length > 2 ? 0 : 14;
    for (const { l, ins } of ops) {
      const sim = l === "aaq_exact0" ? "*" : "";
      if (lineH) {
        g.push(`<text class="op" style="fill:${sv(ins.slot)}" x="${b.x + 10}" y="${y}">${esc(clip(ins.op + sim, maxc))}</text>`);
        g.push(`<text class="args" x="${b.x + 10}" y="${y + 13}">${esc(clip(ins.args.join(" "), maxc + 1))}</text>`);
        y += 32;
      } else {
        g.push(`<text class="op" style="fill:${sv(ins.slot)}" x="${b.x + 10}" y="${y}">${esc(clip(ins.op + " " + ins.args.join(" "), maxc))}</text>`);
        y += 16;
      }
    }
    // register chips, flowing from bottom
    let cx = b.x + 8, cy = BY + BH - 26;
    const rows = [];
    for (const ch of b.chips) {
      const w = ch.length * 6.4 + 12;
      if (cx + w > b.x + b.w - 6) { cx = b.x + 8; cy -= 22; }
      rows.push(`<g class="chip${chg.has(ch) && c ? " chg" : ""}"><rect x="${cx}" y="${cy}" width="${w}" height="18" rx="3"/><text x="${cx + w / 2}" y="${cy + 12.5}" text-anchor="middle">${ch}</text></g>`);
      cx += w + 4;
    }
    g.push(...rows, "</g>");
    parts.push(g.join(""));
  }
  // XMEM bar + paths
  const last = DP_BLOCKS[DP_BLOCKS.length - 1];
  const x0 = DP_BLOCKS[1].x, x1 = last.x + last.w;
  parts.push(`<rect class="xm" x="${x0}" y="${XY}" width="${x1 - x0}" height="${XH}" rx="6"/><text class="bt" x="${x0 + 12}" y="${XY + 20}">XMEM</text><text class="bs" x="${x0 + 12}" y="${XY + 36}">external memory · rows of 128 elements, each with a metadata word</text>`);
  const ag = DP_BLOCKS[1], mu = DP_BLOCKS[2], aq = last;
  const ld = !!lm.load0;
  const lut = !!lm.aaq0 && lm.aaq0.op === "LD.LUT";  // the AaQ slot's own XMEM read
  const quant = !!lm.aaq0 && !lut, exact = !!lm.aaq_exact0;
  const rdTxt = reads.length ? `${fmtBytes(reads.reduce((a, m) => a + m[2], 0))} @ row ${Math.floor(reads[0][1] / 128)}` : "";
  parts.push(edge(ld, "load", arrow(ag.x + ag.w / 2, BY + BH, ag.x + ag.w / 2, XY), ld ? "address" : "", ag.x + ag.w / 2 + 5, BY + BH + 30, "start"));
  parts.push(edge(ld, "load", arrow(mu.x + 40, XY, mu.x + 40, BY + BH), ld ? "row + metadata " + rdTxt.replace(/ @ row.*/, "") : "load row", mu.x + 46, BY + BH + 30, "start"));
  parts.push(edge(lut, "aaq", arrow(aq.x + 40, XY, aq.x + 40, BY + BH), lut ? "LUT segment " + rdTxt.replace(/ @ row.*/, "") : "LD.LUT", aq.x + 46, BY + BH + 30, "start"));
  const wTxt = writes.length ? fmtBytes(writes.reduce((a, m) => a + m[2], 0)) : "store row";
  parts.push(edge(quant, "aaq", arrow(aq.x + aq.w / 2 + 30, BY + BH, aq.x + aq.w / 2 + 30, XY), quant ? `${wTxt} + metadata` : "store row", aq.x + aq.w / 2 + 35, BY + BH + 30, "start"));
  parts.push(edge(exact, "aaq_exact", arrow(aq.x + aq.w - 30, BY + BH, aq.x + aq.w - 30, XY), exact ? `${wTxt}*` : "", aq.x + aq.w - 25, BY + BH + 30, "start", " sim"));
  parts.push(`<text class="bs" x="${x1}" y="${XY + XH + 14}" text-anchor="end">* simulation-only slot, not in hardware (IPU-ARCH-4)</text>`);
  $("dp").innerHTML = parts.join("");
}

const STAGES = [
  { k: "CTRL", col: "lr", slots: ["cond", "lr", "break"] },
  { k: "Cache agent", short: "AGT", col: "load", slots: ["load"] },
  { k: "MULT", col: "mult", slots: ["mult"] },
  { k: "ACC", col: "acc", slots: ["acc"] },
  { k: "AAQ", col: "aaq", slots: ["aaq", "aaq_exact"] },
];
const stageOps = (pc, st) => T.program[pc].slots.filter((s) => st.slots.includes(s.slot));
function renderTiming() {
  const N = T.cycles.length, clk = pos - 1; // newest word (pos-1) is in CTRL at clock clk
  $("stages").innerHTML = STAGES.map((st, s) => {
    const w = clk - s;
    if (pos === 0 || w < 0) return `<div class="stage" style="--sc:${sv(st.col)}"><span class="sn">${st.k}</span><span class="sb">empty</span></div>`;
    const pc = T.cycles[w].pc, ops = stageOps(pc, st);
    return `<div class="stage${ops.length ? " busy" : ""}" style="--sc:${sv(st.col)}"><span class="sn">${st.k}</span><span class="sw">word #${w} · ${hex4(pc)}</span>${
      ops.length ? ops.map((o) => `<span class="so">${esc(o.op)} <small>${esc(o.args.join(" "))}</small></span>`).join("") : '<span class="sb">bubble</span>'}</div>`;
  }).join("");
  const ROWS = 8;
  const wLast = Math.max(0, clk), wFirst = Math.max(0, wLast - ROWS + 1);
  const t0 = wFirst, t1 = wLast + STAGES.length - 1;
  const ncol = t1 - t0 + 1;
  const cells = [`<div></div>`];
  for (let t = t0; t <= t1; t++) cells.push(`<div class="pc-h${t === clk ? " now" : ""}">${t}</div>`);
  if (pos === 0) { $("pchart").style.gridTemplateColumns = "1fr"; $("pchart").innerHTML = '<div class="hint">Step once to fetch the first word into CTRL.</div>'; return; }
  for (let w = wFirst; w <= wLast && w < N; w++) {
    const pc = T.cycles[w].pc;
    cells.push(`<div class="pc-w${w === clk ? " cur" : ""}" title="${esc(T.program[pc].slots.map((s) => s.op).join(", ") || "all NOP")}">#${w} ${hex4(pc)}</div>`);
    for (let t = t0; t <= t1; t++) {
      const s = t - w;
      if (s < 0 || s >= STAGES.length) { cells.push("<div></div>"); continue; }
      const st = STAGES[s], ops = stageOps(pc, st);
      const cls = `pc-c ${ops.length ? "op" : "bub"}${t > clk ? " fut" : ""}${t === clk ? " now" : ""}`;
      cells.push(`<div class="${cls}" style="--sc:${sv(st.col)}" data-tip="${esc(`word #${w} ${hex4(pc)} in ${st.k} at clock ${t}\n${ops.length ? ops.map((o) => o.op + " " + o.args.join(" ")).join("\n") : "bubble (no op for this stage)"}${t > clk ? "\n(future clock)" : ""}`)}">${st.short || st.k}</div>`);
    }
  }
  $("pchart").style.gridTemplateColumns = `86px repeat(${ncol}, minmax(34px, 1fr))`;
  $("pchart").innerHTML = cells.join("");
}
function renderPipeline() { if (!T) return; pview === "datapath" ? renderDatapath() : renderTiming(); }

/* ---------- timeline ---------- */
let TL_ROWS = [];  // set once isa.json gives the slots
const tl = $("tl");
let tlBase = null;
function slotMask(pc) { const m = new Set(T.program[pc].slots.map((s) => s.slot)); return m; }
function drawTimelineBase() {
  if (!T) return;
  $("tl-labels").style.gridTemplateRows = `repeat(${TL_ROWS.length}, 1fr)`;
  $("tl-labels").innerHTML = TL_ROWS.map((r) => `<span style="color:${r === "xmem" ? "var(--muted)" : r === "macs" ? "var(--ink)" : sv(r)}" ${r === "macs" ? 'title="Active multiplier lanes out of 128"' : ""}>${r === "xmem" ? "XMEM" : r === "macs" ? "MAC LANES" : SLOT_NAME[r]}</span>`).join("");
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const W = tl.clientWidth, H = tl.clientHeight;
  tl.width = W * dpr; tl.height = H * dpr;
  const g = tl.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.fillStyle = COLORS.panel; g.fillRect(0, 0, W, H);
  const N = T.cycles.length, rh = H / TL_ROWS.length;
  const cols = Math.max(1, Math.min(W, N));
  const cw = W / cols;
  g.strokeStyle = COLORS.line; g.lineWidth = 1;
  for (let r = 1; r < TL_ROWS.length; r++) { g.beginPath(); g.moveTo(0, Math.round(r * rh) + .5); g.lineTo(W, Math.round(r * rh) + .5); g.stroke(); }
  const masks = T.program.map((_, i) => slotMask(i));
  for (let x = 0; x < cols; x++) {
    const a = Math.floor((x / cols) * N), b = Math.max(a + 1, Math.floor(((x + 1) / cols) * N));
    const counts = new Array(TL_ROWS.length).fill(0);
    for (let k = a; k < b; k++) {
      const m = masks[T.cycles[k].pc];
      TL_ROWS.forEach((r, ri) => { if (r === "macs") counts[ri] += (T.cycles[k].L || 0) / 128; else if (r === "xmem" ? T.cycles[k].m.length : m.has(r)) counts[ri]++; });
    }
    TL_ROWS.forEach((r, ri) => {
      if (!counts[ri]) return;
      const f = counts[ri] / (b - a);
      if (r === "macs") {  // bar height = fraction of the 128 lanes busy
        g.globalAlpha = 1; g.fillStyle = COLORS["s-mult"];
        const h = (rh - 3) * f; g.fillRect(x * cw, ri * rh + rh - 1 - h, Math.max(1, cw - (cw > 3 ? 1 : 0)), h);
        return;
      }
      g.globalAlpha = 0.25 + 0.75 * f;
      g.fillStyle = r === "xmem" ? COLORS.muted : COLORS["s-" + r];
      g.fillRect(x * cw, ri * rh + 2, Math.max(1, cw - (cw > 3 ? 1 : 0)), rh - 4);
    });
  }
  g.globalAlpha = 1;
  if (breakpoints.size) {
    g.fillStyle = COLORS["s-break"];
    T.cycles.forEach((c, k) => { if (breakpoints.has(c.pc)) g.fillRect((k / N) * W, 0, 1.5, 4); });
  }
  tlBase = g.getImageData(0, 0, tl.width, tl.height);
}
function drawPlayhead() {
  if (!tlBase) return;
  const g = tl.getContext("2d"), dpr = Math.min(2, window.devicePixelRatio || 1);
  g.putImageData(tlBase, 0, 0);
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W = tl.clientWidth, H = tl.clientHeight, N = T.cycles.length;
  const x = (pos / N) * W;
  g.fillStyle = COLORS.ink; g.globalAlpha = 0.12; g.fillRect(0, 0, x, H); g.globalAlpha = 1;
  g.fillStyle = COLORS["cur-line"]; g.fillRect(Math.min(W - 2, x), 0, 2, H);
}
function tlSeek(e) { const r = tl.getBoundingClientRect(); const f = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)); stop(); seek(Math.round(f * T.cycles.length)); }
let dragging = false;
tl.addEventListener("pointerdown", (e) => { if (!T) return; dragging = true; tl.setPointerCapture(e.pointerId); tlSeek(e); });
tl.addEventListener("pointermove", (e) => { if (dragging) tlSeek(e); });
tl.addEventListener("pointerup", () => { dragging = false; });

/* ---------- playback ---------- */
function seek(k) { pos = Math.max(0, Math.min(T.cycles.length, k)); update(); }
function update() {
  if (!T) return;
  const N = T.cycles.length;
  const exec = pos ? T.cycles[pos - 1].pc : null;
  const next = pos < N ? T.cycles[pos].pc : null;
  $("r-cycle").textContent = `${pos} / ${N}`;
  $("r-pc").textContent = next == null ? (T.truncated ? "…" : "halt") : hex4(next);
  let mem = 0; for (let k = 0; k < pos; k++) for (const m of T.cycles[k].m) mem += m[2];
  $("r-mem").textContent = mem >= 1024 ? (mem / 1024).toFixed(1) + " KiB" : mem + " B";
  let lanes = 0; for (let k = 0; k < pos; k++) lanes += T.cycles[k].L || 0;
  $("r-mac").textContent = pos ? Math.round((100 * lanes) / (128 * pos)) + "%" : "—";
  for (const el of document.querySelectorAll(".row.cur, .row.next")) el.classList.remove("cur", "next");
  if (exec != null) { const r = $("row-" + exec); if (r) { r.classList.add("cur"); scrollRow(r); } }
  if (next != null) { const r = $("row-" + next); if (r) r.classList.add("next"); }
  $("b-back").disabled = $("b-reset").disabled = pos === 0;
  $("b-step").disabled = $("b-end").disabled = pos >= N;
  renderPipeline();
  renderVLIW(exec);
  renderScalars();
  renderVectors();
  renderEffects();
  drawPlayhead();
}
function scrollRow(r) {
  const box = $("listing"), top = r.offsetTop - box.offsetTop, bot = top + r.offsetHeight;
  if (top < box.scrollTop || bot > box.scrollTop + box.clientHeight) box.scrollTop = top - box.clientHeight / 3;
}
function speed() { const v = +$("speed").value; return Math.round(Math.pow(10, v / 100 * 3)); } // 1..1000 cycles/s
function speedLabel() { $("speedlbl").textContent = `${speed()} cyc/s`; }
function play(until) {
  if (timer) return stop();
  if (pos >= T.cycles.length) pos = 0;
  setRun(true);
  let acc = 0, last = performance.now(), first = true;
  const tick = (now) => {
    acc += (now - last) / 1000 * speed(); last = now;
    let steps = Math.floor(acc); acc -= steps;
    if (first) { steps = Math.max(steps, 1); first = false; }
    while (steps-- > 0) {
      if (pos >= T.cycles.length) { stop(); break; }
      pos++;
      const nx = T.cycles[pos];
      if ((nx && breakpoints.has(nx.pc)) || (until && until(pos))) { stop(); break; }
    }
    update();
    if (timer) timer = requestAnimationFrame(tick);
  };
  timer = requestAnimationFrame(tick);
}
function stop() { if (timer) cancelAnimationFrame(timer); timer = null; setRun(false); }
function setRun(on) {
  $("runlbl").textContent = on ? "Pause" : "Run";
  $("runicon").innerHTML = on ? '<path d="M3 2h4v12H3zM9 2h4v12H9z"/>' : '<path d="M4 2v12l10-6z"/>';
}
function runUntil(pred) {
  stop();
  for (let k = pos + 1; k <= T.cycles.length; k++) if (pred(k)) { seek(k); return; }
  seek(T.cycles.length);
}
const isBranchTaken = (k) => k < T.cycles.length && T.cycles[k].pc !== T.cycles[k - 1].pc + 1;
const isStore = (k) => T.cycles[k - 1].m.some((m) => m[0] === "W");

$("b-run").addEventListener("click", () => play());
$("b-step").addEventListener("click", () => { stop(); seek(pos + 1); });
$("b-back").addEventListener("click", () => { stop(); seek(pos - 1); });
$("b-reset").addEventListener("click", () => { stop(); seek(0); });
$("b-end").addEventListener("click", () => { stop(); seek(T.cycles.length); });
$("b-branch").addEventListener("click", () => runUntil(isBranchTaken));
$("b-store").addEventListener("click", () => runUntil(isStore));
$("speed").addEventListener("input", speedLabel);
document.addEventListener("keydown", (e) => {
  if (!T || e.target.closest("input, select, textarea")) return;
  if (e.key === "Escape") return closeDoc();
  if (e.key === "ArrowRight") { e.preventDefault(); stop(); seek(pos + 1); }
  else if (e.key === "ArrowLeft") { e.preventDefault(); stop(); seek(pos - 1); }
  else if (e.key === " " && !e.target.closest("button")) { e.preventDefault(); play(); }
  else if (e.key === "Home") { stop(); seek(0); }
  else if (e.key === "End") { stop(); seek(T.cycles.length); }
});

/* ---------- tabs ---------- */
for (const t of document.querySelectorAll(".tab")) t.addEventListener("click", () => {
  for (const x of document.querySelectorAll(".tab")) x.setAttribute("aria-selected", x === t);
  for (const v of ["inspector", "source", "isa"]) $("view-" + v).hidden = v !== t.dataset.view;
  if (t.dataset.view === "inspector") { drawTimelineBase(); update(); }
});

/* ---------- theme + resize ---------- */
function retheme() { readColors(); if (T) { drawTimelineBase(); update(); } }
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", retheme);
new MutationObserver(retheme).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
let rz; addEventListener("resize", () => { clearTimeout(rz); rz = setTimeout(() => { if (T) { drawTimelineBase(); drawPlayhead(); } }, 120); });

/* ---------- boot ---------- */
function renderKernelButtons() {
  $("kernels").innerHTML = [["hardware", "Hardware formats"], ["emulator", "Emulator-only formats"]].map(([g, label]) =>
    `<div class="kgroup"><span class="eyebrow">${label}</span><div class="kbtns">${KERNELS.filter((k) => k.group === g).map((k) =>
      `<button class="kbtn" role="tab" data-id="${esc(k.id)}" aria-selected="false">${esc(k.title)} <span class="kdt">${esc(k.label)}</span><small>${k.total_cycles.toLocaleString()} cyc</small></button>`).join("")}</div></div>`).join("");
}
$("kernels").addEventListener("click", (e) => { const b = e.target.closest(".kbtn"); if (b) loadKernel(b.dataset.id); });
speedLabel();
getJSON("data/kernels.json").then((list) => {
  if (list.schemaVersion !== KERNELS_SCHEMA) throw new Error(`kernels.json version ${list.schemaVersion}, viewer ${KERNELS_SCHEMA}`);
  if (!list.kernels.length) throw new Error("kernels.json lists no kernel");
  KERNELS = list.kernels;
  renderKernelButtons();
  const saved = store.get("ipu-kernel");
  return loadKernel(KERNELS.some((k) => k.id === saved) ? saved : KERNELS[0].id);
}).then(() => {
  // open on a working state: first cycle that issues a MULT
  if (T && pos === 0) { const k = T.cycles.findIndex((c) => T.program[c.pc].slots.some((s) => s.slot === "mult")); if (k >= 0) seek(k + 1); }
}).catch((e) => {
  $("listing").innerHTML = `<li class="err">Could not load the kernel list: ${esc(e.message)}</li>`;
});
})();
