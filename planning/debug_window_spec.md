# Debug Window — Build Spec (for Fable 5)

## Prompt to hand Fable

> Read `planning/debug_window_spec.md` in full, then read the files it references in the "Reference: exact emulator structures" section directly from source (don't take the spec's line numbers as gospel — they may drift) before writing any code. Build what the spec describes. Do not explain your reasoning back to me before or after — just build it, and tell me when it's done and how to run it.

Keep this prompt short. The spec below is the actual content; don't inline more of it into the message than the pointer above — Fable reads the file and the referenced source directly.

## Why

This emulator's hardest bugs are VLIW execution-order surprises, not logic errors — a `mult.ve` silently reading a stale snapshot value instead of live, or an XMEM write becoming visible to a same-cycle MULT read. These have recurred as separate incidents (mult Ra/Rc snapshot semantics, xmem-visible-to-mult, str_acc_reg+incr ordering — see project memory) because the current debug loop (`debug_cli.py` register dumps + manual reasoning) doesn't make live-vs-snapshot state visible at a glance.

Beyond debugging, this is also meant as an **explanation tool**: a way to show someone who didn't write this codebase how an assembly command maps to what actually happens in hardware — PC → decoded VLIW word → which registers get touched, in what order, from which state (live or snapshot). Both the visual and text views should read naturally to a newcomer tracing through a program, not just to someone already fluent in the codebase's internals.

## Goal

An interactive cycle-by-cycle debug window for the IPU emulator, in two rendering modes:

1. **Visual mode** — human-facing, interactive stepper (step forward/back through cycles, watch state update live). No LLM involvement.
2. **Text mode** — same underlying state, rendered as a deterministic structured text/JSON dump. No visuals, no LLM involvement. Meant to be pasted into a chat transcript for debugging discussions.

Both modes read from the **same state snapshot object** (see below) — they are two renderers over one data model, not two separate tools.

Out of scope for v1: any LLM/API call from inside the tool. See the appendix for a sketched future extension (Haiku-assisted analysis) — do not build it now.

---

## What must be visible

Per cycle, for the currently-selected point in execution:

- **Program counter** and the decoded instruction at that PC (all active VLIW slots this cycle: `break`, `load`, `store`, `acc_store`, `mult`, `acc`, `aaq`, `cond`, and up to 3 `lr` sub-slots)
- **Full register state**: LR (16×u32), CR (16×u32, note CR15 reserved), R0/R1 (2×128B), r_cyclic (512B), r_mask (128B / 8×16B slots), r_acc (512B / 128×u32 word view), mult_res (512B / 128×u32 word view), post_aaq_reg
- **XMEM read/write address** for any load/store this cycle — the actual byte address, not just the operand register holding it
- **r_cyclic read**: which 128B window was read this cycle (exact byte range, e.g. `r_cyclic[128:256]`), sourced from `ldr_cyclic_mult_reg`'s `offset+base` computation
- **R0/R1 read for mult**: whether the whole 128B vector was used or a single broadcast element (`fixed_ra_idx`), and which one (r0 vs r1)
- **Applied mask for each multiplication**: the actual effective 128-bit mask after mask_offset/mask_shift/partition-vector logic is applied — not just the raw mask register slot. Must be computed via the same logic as `Ipu._mult_mask_and_shift`, since mask_shift can transform the base 128-bit mask before it's applied.
- **Live vs snapshot tag** on every read: whether an operand was resolved from `self.snapshot` or from `state.regfile` (live), since this is the source of the recurring class of bugs in this codebase (mult Ra/Rc reads snapshot, ACC's r_acc reads snapshot, LR/XMEM/MULT writes visible same-cycle to later slots — see `feedback_xmem_writes_visible_to_mult.md` and `mult_ra_rc_snapshot_semantics` in project memory)

---

## Data model (build this first — both renderers consume it)

A single per-cycle snapshot struct, something like:

```python
@dataclass
class CycleSnapshot:
    cycle: int
    program_counter: int
    slots: dict[str, SlotExecution]   # keyed by slot type ("mult", "load", "lr0", "lr1", "lr2", ...)
    regfile: dict[str, bytes]         # full register dump, live state AFTER this cycle
    regfile_snapshot: dict[str, bytes]  # the snapshot state used for reads THIS cycle (before-cycle values)

@dataclass
class SlotExecution:
    slot_type: str
    instruction_name: str | None      # None if slot idle this cycle
    operands: dict[str, OperandRead]

@dataclass
class OperandRead:
    name: str
    value: int | bytes
    source: Literal["live", "snapshot", "immediate"]
    # for register operands: which register + index/address was actually touched
    register: str | None
    address_or_index: int | tuple[int, int] | None  # e.g. xmem byte address, or r_cyclic (start, len)
```

Concrete per-instruction fields to populate (from the actual emulator, not guessed):

- `execute_ldr_cyclic_mult_reg`: `addr = offset + base` (XMEM read address), `index` (r_cyclic write slot, must be multiple of 128)
- `execute_mult_ve`: `ra_idx` → resolve which of r0/r1 + whether `fixed_ra_idx` selects a single broadcast byte or the whole 128B vector is used; `cr_idx` (live CR read); mask fields → effective mask (see below)
- `execute_acc_add` / `acc.*` family: `mult_res` (live read), `r_acc` (**snapshot** read — this is the one that bites people)

**Effective mask computation**: replicate `Ipu._mult_mask_and_shift` steps 1–4 as a pure function (`mask_offset, mask_shift, dstructure_cr_idx, r_mask_bytes) -> int` (128-bit mask), without executing step 5 (the actual zeroing). This lets the debug tool show "here's the mask that will be applied" without mutating state.

**How to drive the emulator**: step `Ipu.execute_vliw_cycle()` one cycle at a time (this already takes the snapshot internally at the top of the cycle — the tool should capture `state` before and after each call, plus snapshot the pre-cycle regfile itself if it needs to show snapshot-vs-live diffs operand-by-operand). Reuse `state.inst_mem[pc]` + `_INSTRUCTION_FIELD_MAP` + `Ipu._resolve_operand` (see survey below) to decode operands rather than re-parsing instruction encoding.

---

## Renderer 1: Visual (interactive stepper)

- Step forward / step back through captured `CycleSnapshot`s (buffer them as the program runs, or re-run deterministically to seek — emulator is deterministic given the same input program+data, so backward stepping can just replay from cycle 0 to N-1 if buffering the whole run is too memory-heavy)
- Views: PC + disassembly of current VLIW word (all slots), register panel (LR/CR/R0/R1 collapsed by default, expandable), r_cyclic as a wrapped ring (mark the current window read), mask panel showing raw mask register + the computed effective mask side by side, XMEM address of any load/store this cycle
- Suggest a TUI (e.g. `textual` or `curses`) or a lightweight local web view — pick whichever is faster to build; no strict requirement either way since this is Fable's implementation call, not something to over-specify here
- Highlight live-vs-snapshot reads distinctly (e.g. color-coded) since that's the primary debugging value-add

## Renderer 2: Text/chat dump

- Given a `CycleSnapshot` (or a cycle range), emit a deterministic structured text block — plain enough to paste into a chat window and read without tooling
- Suggested format: one line per active slot per cycle, e.g.
  ```
  cycle 142 pc=88
    mult: mult.ve ra=r0[fixed_idx=12] cr=cr3(live)=0x00A1 mask_offset=lr8(live)=2 mask_shift=lr9(live)=1 -> effective_mask=0xFF00...(128b) mult_res[live-write]
    acc:  acc.add mult_res(live)=... r_acc(SNAPSHOT)=... -> r_acc(live-write)=...
    lr0:  set lr3 5
    load: ldr_cyclic_mult_reg offset=lr5(live)=256 base=cr2(live)=0x1000 -> xmem_addr=0x1100 (128B) -> r_cyclic[256:384]
  ```
- No prose, no LLM formatting — just a faithful text serialization of the same `CycleSnapshot` struct. Keep it compact (this is meant to be pasted, not scrolled through).

---

## Reference: exact emulator structures (from repo survey, do not re-derive)

- **`IpuState`** — `src/tools/ipu-emu-py/src/ipu_emu/ipu_state.py:40` — `regfile`, `xmem`, `program_counter`, `inst_mem` (list[dict], len 1024), `is_halted`. `IpuState.to_dict()` (line 166) is an existing precedent for state export.
- **`RegFile`** — `regfile.py:42` — backing store `self._storage: dict[str, bytearray]`. Generic access: `regfile.get_register_bytes(name, index)` (line 147), `regfile.raw(name)` (line 93, live bytearray). Cyclic reads: `get_r_cyclic_at(start_idx, length=128)` (line 352).
- **Register shapes** — `ipu-common/src/ipu_common/registers.py:37` (`REGISTER_DEFINITIONS`): `r` (2×128B, r0/r1), `r_cyclic` (512B, wraps), `r_mask` (128B/8×16B slots), `r_acc` (512B, word_view→128×u32), `mult_res` (512B, word_view), `post_aaq_reg` (512B), `lr`/`cr` (16×u32 each, CR15 reserved), `mem_bypass` (128B).
- **`XMem`** — `xmem.py:25` — `self._data = bytearray(1<<21)`; `read_address(addr, size)` / `write_address(addr, data)`.
- **Handlers** (`ipu-emu-py/src/ipu_emu/ipu.py`, class `Ipu`):
  - `execute_ldr_cyclic_mult_reg(self, *, offset, base, index)` — line 461
  - `execute_mult_ve(self, *, ra_idx, cr_idx, mask_offset, mask_shift, dstructure_cr_idx)` — line 705 (Ra from `self.snapshot.raw("r")`, CR from live)
  - `execute_acc_add(self)` — line 784 (`mult_res` live, `r_acc` from `self.snapshot.raw("r_acc")`)
  - `Ipu._mult_mask_and_shift(self, mask_idx, shift, cr_idx)` — line 361 (the authoritative effective-mask computation; steps 1–4 are the pure part to replicate)
  - `Ipu._resolve_operand` — line 278 (central live-vs-snapshot resolution per operand, driven by `"read": "snapshot"|"live"` in `instruction_spec.py`)
  - `Ipu.execute_vliw_cycle` — line 1217 (snapshot taken at line 1235; dispatch order: `break` → LR ×3 → `load` → `mult` → `acc` → `aaq` → `store` → `acc_store` → `cond`)
- **Existing debug precedent**: `debug_cli.py` — `DebugCLI(cmd.Cmd)` (line 295), GDB-like REPL (`do_continue`, `do_step`, `do_regs`, `do_get`, `do_getw`, `do_set`, `do_disasm`, `do_save`); `emulator.py` — `run_with_debug(state, debug_callback, max_cycles)` (line 86), callback returns `CONTINUE`/`STEP`/`QUIT`. No existing cycle-by-cycle trace accumulator — the new tool must build one by stepping `execute_vliw_cycle()` itself.
- **Instruction decode**: `state.inst_mem[pc]` is a decoded field dict. `ipu_common.instruction_spec.get_instruction_by_opcode(slot_type, opcode)` (`instruction_spec.py:1178`) → `(name, spec_dict)`; `spec_dict["operands"]` is `[{"name", "type", "read"?}]`. Reuse `Ipu`'s precomputed `_INSTRUCTION_FIELD_MAP[(slot_type, inst_name)]` to map operand name → raw field key, then `_resolve_operand` for actual values. Slot types: `break`, `load`, `store`, `acc_store`, `mult`, `acc`, `aaq`, `cond`, `lr` (3 sub-slots).

---

---

## v2 addendum: vector register layout & dtype awareness (visual mode only)

### Prompt to hand Fable (v2)

> The debug window v1 is built and working (`debug_window_html.py`, `debug_window.py`). Read the "v2 addendum" section of `planning/debug_window_spec.md` in full, then read the current `debug_window_html.py` and `debug_window.py` directly from source before changing anything — the spec's line numbers may drift. Implement only what the v2 addendum describes; v1 behavior (slot chips, live/snapshot tags, mask panel, r_cyclic ring, text/JSON renderers) must keep working exactly as it does now. Do not explain your reasoning back to me — just build it, and tell me when it's done and how to run it.

### Why (v2)

Two gaps found using v1 against `conv_universal_bn_activation`: (1) R0/R1 render as a flat undifferentiated hex dump while r_cyclic gets a purpose-built row layout — inconsistent, and it makes comparing "what mult is about to read from R0/R1" against "what it's about to read from r_cyclic" harder than it should be. (2) every vector register renders as raw hex regardless of what the bytes actually mean — R0/R1/r_cyclic are dtype-coded (INT8 or one of the FP8 e/m variants, per `state.dtype`), while r_mask is not data at all, it's 8 independent 128-bit bitmasks. Raw hex hides this; a newcomer reading the trace to understand hardware behavior (see the top-level "Why") has no way to tell "this byte is -12" from "this byte is a mask bit pattern" without already knowing the codebase.

### Scope

**1. r_cyclic and R0/R1 as row-structured strips, not flat hexdump.**
- Reshape the existing r_cyclic ring canvas (`renderRing()` in `debug_window_html.py`) from one 512-wide horizontal strip into 4 stacked 128-wide rows (one row per 128B chunk). Keep the existing read/write window highlighting (orange = mult read window, blue = load write target) — just wrap it across 4 rows instead of one long strip.
- Give R0 and R1 the same visual treatment: each rendered as a byte-strip matching the row style used for r_cyclic (so a 128B r_cyclic window and a 128B R0/R1 vector look visually comparable), rather than falling into the generic `renderVectors()`/`hexdump()` path they use today. Pull R0/R1 out of the `VECTORS` list's generic-hexdump handling and give them dedicated rendering next to (or reusing) the r_cyclic ring code.

**2. Dtype-aware decoding for R0/R1 and r_cyclic.**
- `state.dtype` (an emulator-wide `DType.INT8` or one of `DType.E1..E7`, see `ipu_math.py`) governs how every byte in R0/R1/r_cyclic should be interpreted. Thread `state.dtype` into the per-cycle trace payload (it isn't captured on `CycleSnapshot` today — add it) so the renderer knows the mode for that specific cycle without assuming it's constant for the whole trace.
- For each byte in R0/R1/r_cyclic's row-strip view, show the decoded value (signed int8 for `DType.INT8`; the float value via `fp8_bytes_to_fp32(raw_byte, dtype)` from `ipu_math.py` for FP8 modes) as the primary label, with raw hex always available (e.g. on hover/tooltip) as a fallback — don't lose the raw-byte view, just don't make it the only view.
- Do not reimplement FP8 decoding — reuse `fp8_bytes_to_fp32` (`ipu_emu/ipu_math.py`) rather than writing new bit-twiddling logic for exponent/mantissa extraction.

**3. r_mask as 8 bitmask strips, not hexdump.**
- Reuse the existing `maskRow()` bit-grid renderer (already used in the per-cycle "Mult mask" panel) for r_mask's full-register view in `renderVectors()`. Render all 8 slots (16 bytes = 128 bits each) as 8 separate `maskRow()` strips, always expanded (not behind a collapsed `<details>` toggle like the other generic vectors) — the point is that this register's structure (8 independent bit-vectors) should be visually obvious without needing to open anything.
- Pull `r_mask` out of the generic `VECTORS` hexdump list the same way R0/R1 are pulled out for #1/#2.

### Do not (v2, in addition to the v1 "Do not" section above)

- Do not change r_acc or mult_res rendering — they're fixed int32 word registers regardless of `state.dtype` (`word_view=True` in `REGISTER_DEFINITIONS`), the existing word-view hexdump is correct and out of scope here.
- Do not change anything about how the emulator itself computes or stores state, or how `state.dtype` is set — this is a rendering-only change against the existing trace-capture pipeline (`debug_window.py`'s `capture_trace`/`CycleSnapshot`/`snapshot_to_dict`) and the HTML renderer (`debug_window_html.py`). If `CycleSnapshot` needs a new field (dtype), add it there — don't touch `ipu.py`/`regfile.py`/`ipu_state.py`/`ipu_math.py` beyond calling the existing public `fp8_bytes_to_fp32`.
- Do not change the text-mode renderer (`debug_window_text.py`) or JSON output shape beyond adding the dtype field needed for #2 — this addendum is visual-mode-only per the original scope split (visual = human GUI, text = plain dump for chat).
- Do not regress v1 behavior: the slot chips, live/snapshot color coding, mask panel for the active cycle's mult, and existing r_cyclic read/write highlighting must all still work after this change.

## Appendix: future extension (not v1, sketch only)

A Haiku-backed assistant layer could sit on top of the `CycleSnapshot` stream to explain anomalies in natural language — e.g. flag "this mult reads r_acc from snapshot while the same cycle's acc.add just wrote it live, matching the pattern in `feedback_xmem_writes_visible_to_mult.md`" — using Haiku for cost/speed since this is pattern-matching against a small fixed set of known bug shapes, not deep reasoning. This would consume the same text-dump format as Renderer 2, so no rework of the data model would be needed if built later. Do not implement this now — v1 is fully deterministic, no API calls.

---

## Do not

- Do not modify `ipu.py`, `regfile.py`, `xmem.py`, `instruction_spec.py`, `registers.py`, or any other emulator/assembler/common source file. This tool is read-only against emulator state — it observes by driving `execute_vliw_cycle()` and inspecting `state`/`regfile`/`snapshot`, it does not change how the emulator executes.
- Do not alter, "clean up", or refactor the VLIW dispatch order, the snapshot mechanism, or any existing `execute_*` handler, even if something looks improvable. The snapshot/live semantics are the exact thing this tool exists to make visible — changing them defeats the purpose and risks silently fixing (and thereby hiding) real bugs this tool should instead surface.
- Do not add a live-vs-snapshot "fix" or "normalize" mode. The tool's job is to show the current behavior accurately, not to change it or offer an opinion on whether it's correct.
- Do not build the Haiku/LLM extension in this pass — appendix only, v1 is fully deterministic.
- Do not add new instructions, new register types, or extend `instruction_spec.py` to support the tool. If an existing instruction's decode path is insufficient, extend the debug tool's own decoding logic, not the shared instruction spec.
- Do not restructure `debug_cli.py` or `emulator.py` — treat them as reference precedent for state-export patterns, not as a foundation to build on top of or modify in place. The new tool can be a separate module that reuses their patterns.
