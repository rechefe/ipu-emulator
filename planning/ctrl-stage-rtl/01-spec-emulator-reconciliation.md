# Reconcile `stage-control.md` against the emulator before any RTL is written

Part of the [CTRL-stage RTL epic](00-epic.md). Blocks everything else.

## Goal

Produce a single, agreed-upon CTRL-stage contract by walking every
hardware-classified slot/opcode the emulator actually dispatches
(`is_hardware_slot(...)` in `instruction_spec.py`) against what
`stage-control.md` describes, and closing every gap — either by updating the
doc (spec was incomplete/wrong) or by filing a follow-up decision (emulator
under-constrains something hardware must constrain). This is a documentation
and decision issue, not an RTL issue — it produces no Verilog.

## Known gaps to resolve (starting list; do not assume this is complete)

1. **`break` slot + cond-slot `BKPT` have no home in the spec.**
   `SLOT_METADATA["break"]` is not marked `hardware: False`, and
   `Ipu.execute_vliw_cycle` dispatches it *before* every other slot,
   short-circuiting all side effects for the cycle on `BreakResult.BREAK`
   (`ipu.py:1237-1240`). Decide:
   - Does CTRL own break/halt evaluation, or does it belong to a debug
     subsystem outside CTRL's boundary? (Given it gates LR/dispatch-bus side
     effects same-cycle, it is hard to place anywhere but CTRL.)
   - What is the external effect — a new output port (e.g. `halted`,
     `break_hit`), a CPU-visible status bit via APB, both?
   - Update §3 (interfaces), §4 (`BRANCH_COND_COUNT` should become 6, or
     `BKPT` should be pulled out of the cond mnemonic count with its own
     line), and add a `§10.4 BREAK Slot` / `§10.2.6 BKPT` section mirroring
     the existing per-instruction format.
2. **`BR` target width.** Spec says "low `PC_W` bits" (7 bits); emulator's
   `execute_br` does not mask. Decide the RTL truncation width (presumably
   `PC_W = 7` per §6.3) and add a note to the emulator issue tracker (or fix
   the emulator directly, see below) so parity testing has a defined,
   in-range BR target space to draw test vectors from.
3. **IMEM sizing/banking.** Spec: `IMEM_DEPTH = 256`, 2×128-entry banked,
   APB-swapped. Emulator: `INST_MEM_SIZE = 1024`, flat, unbanked, no APB.
   Already called out in `riscv-host-integration.md` §4. This is now
   corroborated at the *encoding* level, not just the emulator's array
   size: evaluating `ipu_common.union_layout.get_operand_type_bits()`
   directly shows the cond slot's `Label` operand is **10 bits** wide
   ("`(MAX_PROGRAM_SIZE - 1).bit_length()` for size 1024"), i.e. the
   assembler encodes branch targets for a 1024-word address space, not the
   spec's 256-word/`PC_W = 7` hardware design. The mismatch isn't only
   "the emulator's array happens to be bigger" — the binary format itself
   disagrees with the hardware spec's addressing width. Decide for *this*
   epic: the emulator-parity harness (#2) can only validate ISA-level
   per-instruction semantics and single-active-bank fetch behavior against
   the emulator as-is; banking/APB-swap timing (#6) has no emulator oracle
   until the [RISC-V host integration epic](../riscv-host-integration/00-epic.md)
   lands its MMIO model. Document this boundary explicitly rather than
   pretending #6 is emulator-verified when it can't be yet.
4. **`SET` has no immediate mode.** `stage-control.md` §10.1.1 describes a
   5-bit `src5` operand with a mode-select MSB choosing between a CR read
   and a sign-extended 4-bit immediate (`SET LR2 #-3` in its own example).
   Evaluating the real union layout (`SLOT_UNIONS["lr"]`) shows `SET`'s
   only operand is a plain `CrIdx` field — no mode bit, no immediate path.
   `execute_lr_set` (`ipu.py:511`) is unconditionally `reg = CR[src]`, and
   the assembler's own generated docs (`ipu_as/gen_docs.py:200`) say the
   same thing: "`SET` copies from a `cr` register." Either the immediate
   mode is a real hardware feature nobody has implemented in the assembler/
   emulator yet (in which case it needs an operand type, a union-layout
   entry, and an `execute_lr_set` update before RTL can implement it), or
   it's aspirational prose that was never built and the doc should drop it.
   Decide which, then fix `stage-control.md` §10.1.1 accordingly — do not
   let the RTL silently implement a third, undocumented behavior.
5. **Anything else found during the walk.** Go operand-by-operand through
   every LR-slot and cond-slot instruction's `execute_*` handler versus its
   `stage-control.md` §10 pseudocode (register widths, snapshot vs.
   post-write timing, sign-extension points) and confirm each. The LR/cond
   slots are the two slots CTRL actually implements, so this is the
   complete list — no sampling.

## Suggested resolution for #2 (BR truncation)

Two independent options, not mutually exclusive:
- **Emulator fix:** make `execute_br` mask to the spec's `PC_W` (or to
  `INST_MEM_SIZE` if the emulator's own unbanked address space is treated as
  authoritative pending banking work) so it's a real oracle for BR's
  truncation behavior.
- **RTL-only:** implement the spec's 7-bit truncation regardless, and scope
  BR parity tests to targets that fit in both models' valid ranges — i.e.
  don't claim BR truncation is emulator-verified, just non-contradicted.

Pick one explicitly; don't leave it implicit.

## Deliverables

- [ ] `stage-control.md` updated: `break` slot and `BKPT` fully specified
      (ports, priority, pseudocode, parameter counts corrected).
- [ ] A decision recorded (in the doc or this issue) for `BR` target-width
      handling and whether the emulator changes.
- [ ] A decision recorded for how #6 (APB/banking) will be tested given the
      emulator doesn't model it yet.
- [ ] `stage-control.md` §10.1.1 (`SET`) corrected to match the real,
      CR-only union layout, or the immediate mode is actually implemented
      in the assembler/emulator first — not left contradicting both.
- [ ] A short table, instruction × emulator-handler × spec-section, showing
      every LR-slot and cond-slot opcode has been checked off (this becomes
      the checklist Issues #3/#4/#5 work against).

## Acceptance Criteria

- [ ] No hardware-classified slot or opcode is undocumented in
      `stage-control.md`.
- [ ] Every remaining spec/emulator behavioral difference is either fixed or
      explicitly recorded as an accepted, named gap with a rationale.
- [ ] Issues #3–#5 can cite a spec section for every opcode they implement.
