# CR register file, PC, dual-bank fetch path, and cond/branch resolver

Part of the [CTRL-stage RTL epic](00-epic.md). Depends on #2. Can proceed in
parallel with #3.

## Goal

Implement `ctrl_cr_regfile.sv`, `ctrl_fetch.sv` (PC + `inst mem` + `inst $`),
and `ctrl_cond.sv` (the cond-slot resolver), covering `BEQ`, `BNE`, `BLT`,
`BGE`, `BR`, and — pending #1's resolution — `BKPT`.

## CR register file

- 16×20-bit, 2 banks (`CR_BANKS = 2`), double-buffered like `inst mem`.
- `CR0` hard-wired `0`, `CR1` hard-wired `1` — matches
  `CR_READ_ONLY_INITIAL_VALUES` in `ipu_config.py` and `RegFile._is_read_only_cr`,
  which silently drops writes to those indices rather than erroring; decide
  whether RTL APB writes to `CR0`/`CR1` should instead surface
  `apb_pslverr` per §8 of the spec (the emulator's silent-drop behavior is a
  Python convenience, not necessarily the intended hardware behavior — check
  with #1 if this wasn't already covered there).
- `CR15` dstructure decode (`valid_elements`/`partition`/`pad_mode`) is
  consumed downstream (MULT/ACC/AAQ), not by CTRL itself — CTRL only needs
  to store and forward CR15 like any other CR; don't implement dstructure
  decode logic in CTRL.

## PC and fetch path

Ground truth: `stage-control.md` §6 (this part of the spec is internally
consistent with itself and not obviously behind the emulator, since the
emulator has no PC-width/banking model at all yet — see #1, gap #3). Build
to the spec directly:

- `PC_W = 7` bits (`IMEM_BANK_DEPTH = 128`).
- Dual-port (2R1W per bank) `inst mem`, 2 banks; both `PC+1` and the branch
  target are fetched speculatively every cycle the current instruction is a
  branch (§6.4), so there is never a taken-branch bubble.
- `inst $` is the single-entry pipeline register the controller logic reads
  from; refill mux is `taken ? imem_b : imem_a`.

## Cond/branch resolver

Ground truth for comparison semantics — `ipu.py`, not just the spec prose:

- **`BEQ`/`BNE`** — `execute_beq`/`execute_bne` (`ipu.py:1095-1101`):
  unsigned equality/inequality, no sign handling.
- **`BLT`/`BGE`** — `execute_blt`/`execute_bge` (`ipu.py:1110-1120`): both
  operands go through `_to_signed_reg` (`ipu.py:1103-1108`), which
  sign-extends at **20 bits** (`LR_CR_SCALAR_BITS`), before the signed
  comparison. The RTL comparator must sign-extend from bit 19, not bit 31 —
  this is easy to get wrong if the datapath is modeled as 32-bit anywhere
  upstream.
- **`BR`** — `execute_br` (`ipu.py:1122-1124`): target is the raw register
  value with **no truncation** in the emulator today. Per #1's gap #2,
  confirm the resolved RTL behavior (spec says truncate to `PC_W = 7` bits)
  before finalizing this block, and scope this issue's `BR` parity tests to
  whatever range #1 settled on as jointly valid.
- **`BKPT`** — pending #1. `execute_bkpt` (`ipu.py:1126-1128`) sets
  `program_counter = INST_MEM_SIZE` as an emulator-only "halt" sentinel;
  this is not a real PC value on hardware (`INST_MEM_SIZE` doesn't even
  match `IMEM_BANK_DEPTH`). Do not build this block until #1 defines the
  real hardware behavior (most likely: assert the halt/break port from #5,
  not write a PC value at all).
- **`NOP`** (cond slot) — `execute_cond_nop` (`ipu.py:1130-1132`): plain
  `PC + 1`, i.e. `taken = 0`. This is the same fall-through path every other
  non-branch instruction takes — no special-casing needed if `taken` is
  computed as "cond slot is one of {BEQ,BNE,BLT,BGE,BR} and evaluates true,"
  defaulting to 0 otherwise (`NOP` included).

## Reset

`rst` initializes `PC` to 0 (§3.1) — confirm this matches
`IpuState.__init__`'s `program_counter = 0` (`ipu_state.py:63`, always true
at construction, not just under an explicit reset call — there's no
separate "reset" method on `IpuState` yet, consistent with the direct-Python
harness never needing one; the host-integration epic's soft-reset semantics
in `riscv-host-integration.md` §5 are the closer analog once that lands).

## Tests (via the #2 parity harness)

- [ ] Each of `BEQ`/`BNE`/`BLT`/`BGE`, both branch-taken and fall-through,
      operand pairs spanning the signed 20-bit range including the
      sign-boundary values (`0x7FFFF`/`0x80000`) to catch a sign-extension
      bug at the right bit.
  - [ ] `LR`-vs-`LR`, `LR`-vs-`CR`, and `CR`-vs-`CR` operand combinations
        (both operands are `LcrIdx`).
- [ ] `BR` to several in-range targets (per #1's decision).
- [ ] No-bubble fetch: assert both `imem_read_addr_a`/`imem_read_addr_b`
      fire in the same cycle on a branch instruction, and `inst $` holds the
      correct next instruction at the following cycle regardless of which
      way the branch resolved.
- [ ] `BKPT`, once #1 defines its RTL behavior.

## Acceptance Criteria

- [ ] CR file and fetch path match `stage-control.md` §6 exactly (this part
      of the spec is not in dispute).
- [ ] Cond resolver passes parity on all four comparison branches and `BR`
      against `ipu.py`'s exact sign-extension and comparison semantics.
- [ ] `BKPT` implemented per #1's resolution, not left as a TODO.
