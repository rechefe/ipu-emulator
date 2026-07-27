"""Visual renderer for debug-window traces: a self-contained HTML stepper.

Generates a single HTML file with the whole captured trace embedded as JSON
and a small vanilla-JS UI: step forward/back through cycles, per-slot operand
chips colour-coded live vs snapshot, register panels with change highlighting,
an r_cyclic ring view with the current read window marked, and a mask panel
showing the raw R_MASK slot next to the computed effective mask.

No dependencies, no server, no LLM — open the file in any browser.
"""

from __future__ import annotations

import json
from pathlib import Path

from ipu_emu.debug_window import (
    SLOT_DISPATCH_ORDER,
    CycleSnapshot,
    snapshot_to_dict,
)
from ipu_emu.ipu_math import DType, fp8_bytes_to_fp32


def _fp8_decode_tables(snapshots: list[CycleSnapshot]) -> dict[str, list[float]]:
    """Per-dtype byte→float32 lookup tables for every FP8 mode in the trace.

    Built with the emulator's own ``fp8_bytes_to_fp32`` (single source of FP8
    decoding — no bit-twiddling reimplemented here). INT8 needs no table; the
    JS side does the trivial signed-byte interpretation itself.
    """
    tables: dict[str, list[float]] = {}
    for name in {s.dtype for s in snapshots}:
        dtype = DType[name]
        if dtype == DType.INT8:
            continue
        decoded = fp8_bytes_to_fp32(bytes(range(256)), dtype)
        tables[name] = [float(v) for v in decoded]
    return tables


def trace_to_json(snapshots: list[CycleSnapshot]) -> dict:
    """Compact JSON payload: full 'after' regfile per cycle + one initial
    'before' (cycle N's before-state == cycle N-1's after-state)."""
    cycles = [snapshot_to_dict(s, include_regfile_snapshot=False) for s in snapshots]
    initial = (
        {name: data.hex() for name, data in snapshots[0].regfile_snapshot.items()}
        if snapshots else {}
    )
    return {
        "slot_order": list(SLOT_DISPATCH_ORDER),
        "initial_regfile": initial,
        "decode_tables": _fp8_decode_tables(snapshots),
        "cycles": cycles,
    }


def render_html(snapshots: list[CycleSnapshot], title: str = "IPU debug window") -> str:
    payload = json.dumps(trace_to_json(snapshots), separators=(",", ":"))
    return _TEMPLATE.replace("__TITLE__", title).replace("__PAYLOAD__", payload)


def write_html(snapshots: list[CycleSnapshot], path: str | Path,
               title: str = "IPU debug window") -> Path:
    p = Path(path)
    p.write_text(render_html(snapshots, title), encoding="utf-8")
    return p


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  :root {
    --live:#1f9d55; --snap:#d97706; --imm:#6b7280; --write:#2563eb;
    --bg:#0f172a; --panel:#1e293b; --panel2:#273449; --fg:#e2e8f0; --dim:#94a3b8;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:13px/1.45 "Segoe UI", system-ui, sans-serif; }
  code, .mono { font-family:Consolas, "Cascadia Mono", monospace; }
  header { position:sticky; top:0; z-index:5; background:var(--panel);
           padding:8px 16px; display:flex; gap:12px; align-items:center;
           border-bottom:1px solid #334155; flex-wrap:wrap; }
  header h1 { font-size:15px; margin:0 12px 0 0; }
  button { background:var(--panel2); color:var(--fg); border:1px solid #475569;
           border-radius:5px; padding:4px 12px; cursor:pointer; font-size:13px; }
  button:hover { background:#334155; }
  input[type=number] { width:80px; background:var(--panel2); color:var(--fg);
           border:1px solid #475569; border-radius:5px; padding:4px 6px; }
  .kbd { color:var(--dim); font-size:11px; }
  .legend span { display:inline-block; padding:1px 8px; border-radius:9px;
                 font-size:11px; margin-left:6px; }
  main { display:grid; grid-template-columns: minmax(430px, 1.2fr) 1fr;
         gap:12px; padding:12px 16px; align-items:start; }
  .panel { background:var(--panel); border:1px solid #334155; border-radius:8px;
           padding:10px 12px; margin-bottom:12px; }
  .panel h2 { font-size:12px; text-transform:uppercase; letter-spacing:.08em;
              color:var(--dim); margin:0 0 8px; }
  .slot { border-left:3px solid #475569; padding:6px 8px; margin:6px 0;
          background:var(--panel2); border-radius:0 6px 6px 0; }
  .slot .stype { color:var(--dim); font-size:11px; text-transform:uppercase; }
  .slot .asm { font-weight:600; }
  .chips { margin-top:4px; display:flex; flex-wrap:wrap; gap:4px; }
  .chip { border-radius:10px; padding:1px 8px; font-size:11.5px; cursor:default;
          border:1px solid transparent; font-family:Consolas,monospace; }
  .chip.live { background:rgba(31,157,85,.18); border-color:var(--live); color:#6ee7a8; }
  .chip.snapshot { background:rgba(217,119,6,.18); border-color:var(--snap); color:#fbbf24; }
  .chip.immediate { background:rgba(107,114,128,.18); border-color:var(--imm); color:#cbd5e1; }
  .chip.write { background:rgba(37,99,235,.18); border-color:var(--write); color:#93c5fd; }
  .note { color:var(--dim); font-size:11px; margin-top:2px; }
  table.regs { border-collapse:collapse; width:100%; font-family:Consolas,monospace;
               font-size:12px; }
  table.regs td { padding:1px 6px; white-space:nowrap; }
  td.rn { color:var(--dim); }
  td.changed { color:#f87171; font-weight:700; }
  details { margin:4px 0; }
  summary { cursor:pointer; color:var(--dim); font-size:12px; }
  .hexdump { font-family:Consolas,monospace; font-size:11px; white-space:pre;
             overflow-x:auto; color:#cbd5e1; margin:4px 0; max-height:220px;
             overflow-y:auto; }
  .hexdump .chg { color:#f87171; font-weight:700; }
  .maskrow { display:flex; flex-wrap:wrap; gap:1px; margin:3px 0 8px; }
  .maskrow i { width:7px; height:12px; border-radius:1px; background:#334155; }
  .maskrow i.on { background:var(--live); }
  .maskrow i.chgbit { box-shadow: inset 0 0 0 1px #f87171; }
  .striplabel { color:var(--dim); font-size:11px; margin-top:6px;
                font-family:Consolas,monospace; }
  .masklabel { color:var(--dim); font-size:10px; font-family:Consolas,monospace; }
  .bstrip { display:grid; grid-template-columns:repeat(128,1fr); height:16px;
            margin:2px 0; background:#0b1120; border:1px solid #334155;
            border-radius:3px; }
  .bstrip i { display:block; height:100%; min-width:0; }
  .bstrip i.rd { border-top:3px solid var(--snap); }
  .bstrip i.wr { border-bottom:3px solid var(--write); }
  .bstrip i.chg { box-shadow: inset 0 0 0 1px #f87171; }
  .dtype { background:#1e3a8a; color:#bfdbfe; padding:0 8px; border-radius:9px;
           font-size:11px; font-family:Consolas,monospace; }
  .pcline { font-family:Consolas,monospace; font-size:14px; }
  .tag { padding:0 6px; border-radius:4px; font-size:11px; }
  .tag.brk { background:#7f1d1d; color:#fecaca; }
  .empty { color:var(--dim); font-style:italic; }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <button id="first">&#9198;</button>
  <button id="prev">&#9664; prev</button>
  <button id="next">next &#9654;</button>
  <button id="last">&#9197;</button>
  <span>cycle <input id="jump" type="number" min="0"> / <span id="total"></span></span>
  <span class="kbd">keys: &larr; &rarr; Home End</span>
  <span class="legend">
    <span class="chip live">live read</span>
    <span class="chip snapshot">SNAPSHOT read</span>
    <span class="chip immediate">immediate</span>
    <span class="chip write">write</span>
  </span>
</header>
<main>
  <div>
    <div class="panel">
      <h2>Decoded VLIW word</h2>
      <div class="pcline" id="pcline"></div>
      <div id="slots"></div>
    </div>
    <div class="panel">
      <h2>Mult mask (this cycle)</h2>
      <div id="maskpanel" class="empty">no mult this cycle</div>
    </div>
    <div class="panel">
      <h2 id="stripstitle">Mult-stage data &mdash; R0 / R1 / r_cyclic ring</h2>
      <div class="note">orange top bar = read window this cycle &middot;
        blue bottom bar = load target &middot; red outline = byte changed
        this cycle &middot; hover a cell for the decoded value + raw hex</div>
      <div id="strips"></div>
      <div class="note" id="ringnote"></div>
      <div class="note mono" id="stripreadout">&nbsp;</div>
    </div>
  </div>
  <div>
    <div class="panel">
      <h2>Scalar registers (red = changed this cycle)</h2>
      <table class="regs" id="scalars"></table>
    </div>
    <div class="panel">
      <h2>Vector registers (after this cycle; red bytes = changed)</h2>
      <div id="vectors"></div>
    </div>
  </div>
</main>
<script>
const DATA = __PAYLOAD__;
const N = DATA.cycles.length;
let cur = 0;

function hexToBytes(h){ const a=new Uint8Array(h.length/2);
  for(let i=0;i<a.length;i++) a[i]=parseInt(h.substr(i*2,2),16); return a; }
function u32(bytes,i){ return (bytes[i*4]|(bytes[i*4+1]<<8)|(bytes[i*4+2]<<16)|(bytes[i*4+3]<<24))>>>0; }
function beforeRegs(i){ return i===0 ? DATA.initial_regfile : DATA.cycles[i-1].regfile; }

function esc(s){ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;'); }

function opChip(name, op){
  const cls = op.source;
  let text = name;
  if (op.register==='xmem' && op.address_or_index)
    text += `=xmem@0x${op.address_or_index[0].toString(16)}(${op.address_or_index[1]}B)`;
  else {
    if (op.register) text += `=${op.register}`;
    if (op.address_or_index!=null && op.register!=='xmem'){
      const a=op.address_or_index;
      text += Array.isArray(a) ? `[${a[0]}:${a[1]==null?'':a[0]+a[1]}]` : `[${a}]`;
    }
    if (op.value!=null){
      if (typeof op.value==='object'){
        const h=op.value.hex;
        text += '='+(h.length>16 ? h.slice(0,16)+'..' : h);
      } else text += '='+op.value;
    }
  }
  const tag = op.source==='snapshot' ? ' (SNAPSHOT)' : op.source==='live' ? ' (live)' : '';
  const title = (op.note?op.note+' — ':'')+op.source;
  return `<span class="chip ${cls}" title="${esc(title)}">${esc(text+tag)}</span>`;
}

function writeChip(w){
  let text = w.register;
  if (w.register==='xmem' && Array.isArray(w.address_or_index))
    text = `xmem@0x${w.address_or_index[0].toString(16)}(${w.address_or_index[1]}B)`;
  else if (Array.isArray(w.address_or_index))
    text += `[${w.address_or_index[0]}:${w.address_or_index[1]==null?'':w.address_or_index[0]+w.address_or_index[1]}]`;
  return `<span class="chip write" title="${esc(w.note||'write to LIVE state')}">&rarr; ${esc(text)}</span>`;
}

function renderSlots(c){
  const el = document.getElementById('slots');
  document.getElementById('pcline').innerHTML =
    `PC ${c.pc} &rarr; ${c.pc_after} <span class="dtype">${esc(c.dtype||'INT8')}</span>` +
    (c.break_hit ? ' <span class="tag brk">BREAK</span>':'');
  if (c.empty){ el.innerHTML = '<div class="empty">empty instruction slot (NOP cycle)</div>'; return; }
  let html='';
  for (const key of DATA.slot_order){
    const s = c.slots[key];
    if (!s) continue;
    html += `<div class="slot"><span class="stype">${key}</span> `+
            `<span class="asm mono">${esc(s.assembly)}</span><div class="chips">`;
    for (const [name,op] of Object.entries(s.operands)) html += opChip(name,op);
    for (const w of s.writes) html += writeChip(w);
    html += '</div>';
    const notes=[];
    for (const op of Object.values(s.operands)) if (op.note) notes.push(op.note);
    if (s.mask_note) notes.push('mask: '+s.mask_note);
    if (notes.length) html += `<div class="note">${notes.map(esc).join('<br>')}</div>`;
    html += '</div>';
  }
  el.innerHTML = html || '<div class="empty">all slots NOP</div>';
}

function maskRow(hex){
  // hex = 32 chars, little-bit-order 128-bit int; lane i = bit i
  const v = BigInt('0x'+hex);
  let cells='';
  for (let i=0;i<128;i++){
    const on = (v >> BigInt(i)) & 1n;
    cells += `<i class="${on?'on':''}" title="lane ${i}: ${on?'KEEP':'ZERO'}"></i>`;
  }
  return `<div class="maskrow">${cells}</div>`;
}

function renderMask(c){
  const el = document.getElementById('maskpanel');
  const m = c.slots && c.slots['mult'];
  if (!m){ el.className='empty'; el.textContent='no mult this cycle'; return; }
  el.className='';
  if (m.effective_mask==null){
    el.textContent = m.mask_note || 'mask unavailable'; return;
  }
  el.innerHTML =
    `<div class="note">base mask (raw R_MASK slot) = 0x${m.base_mask}</div>`+
    maskRow(m.base_mask)+
    `<div class="note">effective mask (after shift/partition) = 0x${m.effective_mask}</div>`+
    maskRow(m.effective_mask)+
    `<div class="note">${esc(m.mask_note||'')}</div>`;
}

// ---- dtype-aware byte decoding (FP8 tables come from the emulator's own
// fp8_bytes_to_fp32; INT8 is plain signed-byte interpretation) --------------
function decodeByte(b, dtype){
  if (!dtype || dtype==='INT8') return b<128 ? b : b-256;
  const t = DATA.decode_tables[dtype];
  return t ? t[b] : (b<128 ? b : b-256);
}
function fmtVal(v){
  if (typeof v!=='number') return String(v);
  if (!isFinite(v)) return String(v);
  if (Number.isInteger(v)) return String(v);
  return String(Number(v.toPrecision(4)));
}
function cellColor(b, v){
  if (b===0) return '#141d30';
  if (typeof v!=='number' || !isFinite(v)) return '#a855f7'; // NaN/Inf byte
  const t = Math.min(1, Math.log2(1+Math.abs(v))/8);
  const a = 0.30 + 0.65*t;
  return v<0 ? `rgba(248,113,113,${a})` : `rgba(52,211,153,${a})`;
}

function buildStrip(label, regLabel, aBytes, bBytes, off, idxBase, dtype, rd, wr){
  let cells='';
  for (let i=0;i<128;i++){
    const b=aBytes[off+i], was=bBytes[off+i];
    const v=decodeByte(b,dtype);
    const cls=(rd&&rd.has(idxBase+i)?'rd ':'')+(wr&&wr.has(idxBase+i)?'wr ':'')+
              (b!==was?'chg':'');
    const title=`${regLabel}[${idxBase+i}] = ${fmtVal(v)} · raw 0x${b.toString(16).padStart(2,'0')}`+
      (b!==was?` · changed (was 0x${was.toString(16).padStart(2,'0')})`:'');
    cells+=`<i class="${cls}" style="background:${cellColor(b,v)}" title="${esc(title)}"></i>`;
  }
  return `<div class="striplabel">${esc(label)}</div><div class="bstrip">${cells}</div>`;
}

function renderStrips(c, before){
  const dtype = c.dtype || 'INT8';
  document.getElementById('stripstitle').innerHTML =
    `Mult-stage data &mdash; R0 / R1 / r_cyclic ring, bytes decoded as `+
    `<span class="dtype">${esc(dtype)}</span>`;

  // Highlight sets: r0Rd/r1Rd/r0Wr/r1Wr use local 0..127 indices,
  // rcRd/rcWr use absolute 0..511 r_cyclic indices.
  const r0Rd=new Set(), r1Rd=new Set(), r0Wr=new Set(), r1Wr=new Set();
  const rcRd=new Set(), rcWr=new Set();
  const notes=[];
  if (c.slots && !c.empty){
    const m = c.slots['mult'];
    if (m){
      const rc = m.operands.rc_data;
      if (rc && rc.address_or_index){
        const [start,len]=rc.address_or_index;
        for (let k=0;k<len;k++) rcRd.add((start+k)%512);
        notes.push(`mult read r_cyclic[${start}:${start+len}] (SNAPSHOT data)`);
      }
      for (const opName of ['ra','ra_data','scalar']){
        const op = m.operands[opName];
        if (!op) continue;
        if (op.register==='r0++r1' && op.address_or_index){
          const [start,len]=op.address_or_index;
          for (let k=0;k<len;k++){
            const p=(start+k)%256;
            (p<128 ? r0Rd : r1Rd).add(p%128);
          }
          notes.push(`mult read R0++R1[${start}:${start+len}] (SNAPSHOT data)`);
        } else if (op.register==='r0' || op.register==='r1'){
          const set = op.register==='r0' ? r0Rd : r1Rd;
          if (Array.isArray(op.address_or_index)){
            const [start,len]=op.address_or_index;
            for (let k=0;k<len;k++) set.add((start+k)%128);
            notes.push(`mult read ${op.register}[${start}:${start+len}] (SNAPSHOT data)`);
          } else {
            for (let k=0;k<128;k++) set.add(k);
            notes.push(`mult read whole ${op.register} (SNAPSHOT data)`);
          }
        }
      }
    }
    const l = c.slots['load'];
    if (l){
      for (const w of l.writes){
        if (w.register==='r_cyclic' && Array.isArray(w.address_or_index)){
          const [start,len]=w.address_or_index;
          for (let k=0;k<len;k++) rcWr.add((start+k)%512);
          notes.push(`load wrote r_cyclic[${start}:${start+len}] (LIVE; mult data reads are SNAPSHOT, window index is live)`);
        } else if (w.register==='r0' || w.register==='r1'){
          const set = w.register==='r0' ? r0Wr : r1Wr;
          for (let k=0;k<128;k++) set.add(k);
          notes.push(`load wrote whole ${w.register} (LIVE; not visible to this cycle's mult data reads — SNAPSHOT)`);
        }
      }
    }
  }

  const rA=hexToBytes(c.regfile.r), rB=hexToBytes(before.r);
  const cA=hexToBytes(c.regfile.r_cyclic), cB=hexToBytes(before.r_cyclic);
  let html='';
  html += buildStrip('R0 (128B)','r0', rA,rB, 0, 0, dtype, r0Rd, r0Wr);
  html += buildStrip('R1 (128B)','r1', rA,rB, 128, 0, dtype, r1Rd, r1Wr);
  for (let row=0;row<4;row++){
    html += buildStrip(`r_cyclic[${row*128}:${(row+1)*128}]`,'r_cyclic',
                       cA,cB, row*128, row*128, dtype, rcRd, rcWr);
  }
  document.getElementById('strips').innerHTML = html;
  document.getElementById('ringnote').textContent = notes.join('  |  ');
}
document.getElementById('strips').addEventListener('mouseover', e=>{
  if (e.target.tagName==='I' && e.target.title)
    document.getElementById('stripreadout').textContent = e.target.title;
});

function renderScalars(c, before){
  const el = document.getElementById('scalars');
  const after = c.regfile;
  const groups = [['lr',16],['cr',16]];
  let html='';
  for (const [name,count] of groups){
    const a = hexToBytes(after[name]), b = hexToBytes(before[name]);
    for (let row=0;row<count;row+=4){
      html += '<tr>';
      for (let i=row;i<row+4;i++){
        const av=u32(a,i), bv=u32(b,i);
        const chg = av!==bv;
        const label = name+i + (name==='cr'&&i===15?'*':'');
        html += `<td class="rn">${label}</td><td class="${chg?'changed':''}" title="${chg?('was '+bv):''}">${av}</td>`;
      }
      html += '</tr>';
    }
    html += '<tr><td colspan="8" style="height:6px"></td></tr>';
  }
  el.innerHTML = html;
}

function hexdump(aBytes,bBytes,wordView){
  let out='';
  if (wordView){
    for (let i=0;i<aBytes.length/4;i+=8){
      let line = String(i).padStart(4,' ')+'w  ';
      for (let j=i;j<Math.min(i+8,aBytes.length/4);j++){
        const av=u32(aBytes,j), bv=u32(bBytes,j);
        const h=av.toString(16).padStart(8,'0');
        line += (av!==bv)?`<span class="chg">${h}</span> `:h+' ';
      }
      out += line+'\n';
    }
  } else {
    for (let i=0;i<aBytes.length;i+=32){
      let line = String(i).padStart(4,' ')+'  ';
      for (let j=i;j<Math.min(i+32,aBytes.length);j++){
        const h=aBytes[j].toString(16).padStart(2,'0');
        line += (aBytes[j]!==bBytes[j])?`<span class="chg">${h}</span>`:h;
        if ((j+1)%4===0) line+=' ';
      }
      out += line+'\n';
    }
  }
  return out;
}

// R0/R1 and r_cyclic render as dtype-decoded strips (renderStrips), and
// r_mask as 8 always-expanded bitmask rows below — none of them go through
// the generic hexdump path any more.
const VECTORS = [
  ['r_acc','r_acc (128×u32)',true],
  ['mult_res','mult_res (128×u32)',true],
  ['post_aaq_reg','post_aaq_reg (512B)',false],
  ['mem_bypass','mem_bypass (128B)',false],
];
let openState = {};

function maskRowBytes(bytes, beforeBytes, off){
  let cells='';
  for (let i=0;i<128;i++){
    const on  = (bytes[off+(i>>3)]      >> (i&7)) & 1;
    const was = (beforeBytes[off+(i>>3)]>> (i&7)) & 1;
    cells += `<i class="${on?'on':''}${on!==was?' chgbit':''}"`+
             ` title="lane ${i}: ${on?'KEEP':'ZERO'}${on!==was?' (changed)':''}"></i>`;
  }
  return `<div class="maskrow">${cells}</div>`;
}

function renderMaskReg(c, before){
  const a=hexToBytes(c.regfile.r_mask), b=hexToBytes(before.r_mask);
  let html = '<div class="striplabel">r_mask &mdash; 8 &times; 128-bit slots '+
             '(green = KEEP lane, dark = ZERO lane, red outline = bit changed)</div>';
  for (let s=0;s<8;s++)
    html += `<div class="masklabel">slot ${s}</div>` + maskRowBytes(a,b,s*16);
  return html;
}

function renderVectors(c, before){
  const el = document.getElementById('vectors');
  let html = renderMaskReg(c, before);
  for (const [name,label,wv] of VECTORS){
    if (!(name in c.regfile)) continue;
    const a=hexToBytes(c.regfile[name]), b=hexToBytes(before[name]);
    let changed=0; for(let i=0;i<a.length;i++) if(a[i]!==b[i]) changed++;
    html += `<details data-reg="${name}" ${openState[name]?'open':''}>`+
      `<summary>${label}${changed?` — <span style="color:#f87171">${changed}B changed</span>`:''}</summary>`+
      `<div class="hexdump">${hexdump(a,b,wv)}</div></details>`;
  }
  el.innerHTML = html;
  for (const d of el.querySelectorAll('details'))
    d.addEventListener('toggle',()=>{ openState[d.dataset.reg]=d.open; });
}

function show(i){
  cur = Math.max(0, Math.min(N-1, i));
  const c = DATA.cycles[cur], before = beforeRegs(cur);
  document.getElementById('jump').value = c.cycle;
  document.getElementById('total').textContent =
    `${N} captured (cycles ${DATA.cycles[0].cycle}..${DATA.cycles[N-1].cycle})`;
  renderSlots(c); renderMask(c); renderStrips(c, before);
  renderScalars(c, before); renderVectors(c, before);
}

document.getElementById('next').onclick = ()=>show(cur+1);
document.getElementById('prev').onclick = ()=>show(cur-1);
document.getElementById('first').onclick = ()=>show(0);
document.getElementById('last').onclick = ()=>show(N-1);
document.getElementById('jump').onchange = e=>{
  const want = Number(e.target.value);
  const idx = DATA.cycles.findIndex(c=>c.cycle===want);
  show(idx>=0?idx:cur);
};
document.addEventListener('keydown', e=>{
  if (e.target.tagName==='INPUT') return;
  if (e.key==='ArrowRight') show(cur+1);
  else if (e.key==='ArrowLeft') show(cur-1);
  else if (e.key==='Home') show(0);
  else if (e.key==='End') show(N-1);
});
if (N>0) show(0);
else document.body.innerHTML='<p style="padding:20px">Trace is empty.</p>';
</script>
</body>
</html>
"""
