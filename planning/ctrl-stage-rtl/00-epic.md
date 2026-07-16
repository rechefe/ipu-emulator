# Epic: CTRL-stage RTL, verified against the Python emulator as golden model

## Goal

Implement synthesizable RTL for the **CTRL stage** — the pipeline stage
described in [`docs/content/specs/stage-control.md`](../../docs/content/specs/stage-control.md):
PC + dual-bank instruction memory + `inst $`, the 3-lane LR ALU, the branch/cond
resolver, the internal CR/LR register files, the four dispatch buses
(`mult_vliw_bus`/`acc_vliw_bus`/`aaq_vliw_bus`/`str_vliw_bus`), the XMEM
read-address path, and the host APB slave.

The spec doc is the primary source for the stage's **external contract**
(ports, timing, bus packing). The Python emulator
(`src/tools/ipu-emu-py/src/ipu_emu/ipu.py`, `ipu_state.py`, `execute.py`) is
the primary source for **exact instruction semantics** — bit widths,
truncation/wrap behavior, sign extension, read-before-write ordering — because
it is the tested, executable reference; the spec doc is prose written after
the fact and has already drifted from it in at least the ways listed below.
No RTL for an instruction is considered correct until it is checked
cycle-for-cycle against the corresponding `execute_*` handler, not just
against the prose.

## Why this is two-source, not one

`instruction_spec.py` is the assembler/emulator's single source of truth for
encoding; `ipu.py` is the single source of truth for what each opcode
*computes*. `stage-control.md` is a **third**, independently-written
description of the same instructions at the RTL level, and it is not
generated from either — it's free to say something the emulator doesn't
implement (aspirational hardware behavior) or to omit something the emulator
does implement (an ISA opcode nobody updated the doc for). Treating the doc
as sufficient on its own would silently bake both kinds of drift into the
RTL. Issue #1 below exists specifically to reconcile these before any RTL is
written.

## Known spec/emulator discrepancies (found while scoping this epic)

These are not exhaustive — Issue #1 is the dedicated pass — but three are
already confirmed by reading the code, and shape the rest of this epic:

1. **The `break` slot and the cond-slot's `BKPT` opcode are entirely absent
   from `stage-control.md`.** `instruction_spec.py`'s `"break"` slot
   (`BREAK`, `BREAK.IFEQ`, opcode-position `NOP`) is *not* marked
   `hardware: False` in `SLOT_METADATA` (unlike `acc_store`, which is
   explicitly simulation-only) — so it is a real hardware slot with no
   architectural description anywhere. Worse, `Ipu.execute_vliw_cycle()`
   dispatches the break slot **first**, before any other slot, and a
   `BreakResult.BREAK` return **skips all other side effects for that
   cycle** (`ipu.py:1237-1240`). That priority-gating behavior — a
   same-cycle "does this instruction have side effects at all" decision —
   is CTRL-stage control logic with no home in the current spec: no
   `break`/`halt` port exists in §3, and `BRANCH_COND_COUNT = 5` in §4
   excludes `BKPT` from the cond slot's own opcode count.
2. **`BR`'s target truncation is unspecified in the emulator.**
   `stage-control.md` §10.2.5 says `BR` uses "the low `PC_W` bits" of the
   register (`PC_W = 7`, i.e. `IMEM_BANK_DEPTH = 128`). `Ipu.execute_br`
   (`ipu.py:1122-1124`) does `self.state.program_counter = reg` with **no
   masking at all** — a 20-bit LR/CR value can set a `program_counter` far
   outside a 128-entry bank. The RTL needs a real width for its PC register;
   the emulator, today, does not enforce the one the doc claims.
3. **Instruction memory sizing.** `stage-control.md` specifies
   `IMEM_DEPTH = 256` total across two 128-entry banks (`PC_W = 7`). The
   emulator's `INST_MEM_SIZE = 1024` (`ipu_state.py:28`) is a flat,
   unbanked array with no APB, no bank-swap register, and no host model at
   all — this gap is already called out in
   [`riscv-host-integration.md` §4](../../docs/content/specs/riscv-host-integration.md#4-the-ipu-host-control-register-block)
   ("the emulator uses `INST_MEM_SIZE` (1024) for `IMEM_DEPTH`; hardware
   uses `IMEM_DEPTH = 256`"), so it is a *known*, not newly-discovered, gap —
   but it means the emulator cannot currently serve as a golden model for
   IMEM banking, dual-port fetch races, or APB bank-swap timing. Only the
   ISA-level, single-bank-equivalent behavior is emulator-verifiable today.

None of these block starting the epic, but #1 in particular changes the
CTRL boundary (an added halt/break port) enough that it must be resolved
before the fetch/dispatch RTL (#4/#5) is written, not after.

## Scope

In scope (from `stage-control.md`):

- Internal `PC`, dual-bank `inst mem` (2×128-entry, 2R1W per bank), `inst $`,
  no-bubble speculative dual-fetch on branch cycles.
- Internal CR file (2 banks, `CR0`/`CR1` hard-wired) and LR file (16×20-bit),
  both stage-internal — never exposed on the boundary.
- 3-lane LR ALU (`SET`, `ADD`, `SUB`, `INCR_MOD_POW2`, `INC`, `DEC`) with
  same-cycle write conflict detection.
- Cond/branch resolver (`BEQ`, `BNE`, `BLT`, `BGE`, `BR`) and — pending
  Issue #1 — `BKPT` and the `break` slot's halt-gating.
- Dispatch-bus packing (`mult_vliw_bus`, `acc_vliw_bus`, `aaq_vliw_bus`,
  `str_vliw_bus`) with pre-resolved, post-LR-write CR/LR operands.
- `xmem_read_addr`/`xmem_read_en` resolution.
- APB slave: inactive-bank IMEM/CR writes, bank-swap registers, status.

Out of scope for this epic (tracked elsewhere or by a follow-up):

- MULT/ACC/AAQ/STORE execute-unit RTL (separate stages, separate specs).
- Cycle-accurate host timing — the [RISC-V host integration
  epic](../riscv-host-integration/00-epic.md) models host *sequencing*
  functionally in Python; this epic's APB slave (#6) should reuse whatever
  register-map source of truth that epic produces rather than inventing a
  second one, but is not blocked on that epic's Unicorn/Rust work (#4/#5
  there).
- Physical implementation (timing closure, synthesis constraints, DFT).

## Constraints

- **No RTL instruction lands without an emulator-parity test.** Every
  `execute_*` handler this epic touches gets a corresponding cocotb
  test that feeds the same operands into the RTL and the live `Ipu`
  instance and diffs the results. Prose in `stage-control.md` informs the
  test's *intent*; the emulator's output is the pass/fail oracle.
- **Bazel-hermetic**, matching the rest of the repo — no `pip install`/manual
  simulator invocations outside `bazel test //...`.
- Spec-doc corrections discovered along the way (§ Known discrepancies
  above, and whatever Issue #1 finds) get folded back into
  `stage-control.md` — this epic should leave that doc *more* accurate than
  it found it, not just work around its gaps silently.
- Register-map single-source-of-truth discipline from the rest of the repo
  (`instruction_spec.py`, `registers.py`) extends to the APB register block:
  no hand-duplicated field metadata between RTL, the parity harness, and
  docs.

## Sub-issues

- [ ] #1 — Reconcile `stage-control.md` against the emulator (break/BKPT, PC width, IMEM sizing) before any RTL is written
- [ ] #2 — RTL project scaffold + emulator-parity verification infrastructure
- [ ] #3 — LR register file + 3-lane LR ALU RTL, verified against `execute_lr_*`
- [ ] #4 — CR register file, PC, dual-bank fetch path, and cond/branch resolver
- [ ] #5 — Dispatch-bus packing, XMEM read-address path, and BREAK/BKPT priority gating
- [ ] #6 — APB slave register block (host config + bank swap)
- [ ] #7 — Full CTRL-stage integration, app-level parity suite, sign-off

Dependency order: **1 → 2 → {3, 4} → 5 → 6 → 7**. Issues 3 and 4 can proceed
in parallel once the scaffold (#2) exists; #5 needs both.

## Acceptance Criteria

- [ ] `stage-control.md` accurately describes every hardware-classified slot
      the emulator dispatches (including `break`/`BKPT`), with no unresolved
      discrepancy against `ipu.py`.
- [ ] Synthesizable RTL exists for every block in the CTRL boundary diagram
      (§2 of the spec), each with a passing emulator-parity test suite.
- [ ] A top-level integration test runs at least one real `ipu-apps` program
      (e.g. `fully_connected`) through both the RTL simulation and the Python
      emulator and asserts identical dispatch-bus traces and final register
      state.
- [ ] `bazel test //...` passes with the new RTL toolchain wired in.
- [ ] Any spec inaccuracies found during implementation are fixed in
      `stage-control.md`, not just worked around in code comments.
