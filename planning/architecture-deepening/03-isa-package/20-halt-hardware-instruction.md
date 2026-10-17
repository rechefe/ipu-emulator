# `HALT` is a hardware COND instruction: spec text and RTL control spec

Part of [Epic E03: ISA package](00-epic.md) · ID E03-20 · Size S · Depends on: E03-07, E03-19 · Behaviour change: none in the emulator (docs and ISA text only) · `*_WIDTH` localparams: unchanged (opcode 5 of the 3-bit COND field, as today)

## Goal

Decision Q40: the emulator's `BKPT` becomes the hardware instruction `HALT` (E03-19 renames it). This issue makes the ISA text, the RTL control-stage spec and the requirements agree on what it is: a COND-slot instruction that ends the kernel after every slot of its word has retired.

## Background

- Spec entry (`instruction_spec.py:1135-1145`): title "Breakpoint", summary "Conditional breakpoint", operation "Halt execution (debugging)". The handler is unconditional and runs last in the word: `execute_bkpt` sets `pc = INST_MEM_SIZE` (`ipu_emu/ipu.py:1875-1877`) after every other slot (COND is dispatched last, `ipu.py:2304`). E02-06 `test_bkpt_halts_after_whole_word_retires` pins this.
- RTL control spec: `BRANCH_COND_COUNT = 5` lists `BEQ`, `BNE`, `BLT`, `BGE`, `BR` (`docs/content/specs/stage-control.md:178`); the COND opcode list (`:258`) and §10.2 (`:523-597`) and the summary table (`:598+`) have no halt instruction.
- Requirements already reserve it under another name: IPU-ARCH-5 "The `COND` slot shall provide `END`. `END` uses the opcode that the instruction reference assigns to `BKPT`" (`requirements/architecture.md:87`); IPU-DBG-8 "When the stepped word holds `END`, the kernel is done" (`requirements/debug.md:16`); action A-1 "Replace `BKPT` with `END` ..." (`requirements/open-questions.md:51`). E09-05 asks which issue updates A-1: this one.
- `HALTED` is also the name of a host-visible state (`requirements/execution-flow.md:39`), entered on halt requests, breakpoints, step done and faults.

## Implementation

1. Spec entry for `HALT` (tag HARDWARE from the COND slot, E03-07): title "Halt", summary "End the kernel after every slot of this word has retired.", operation `pc stops; kernel done`, notes: unconditional; the word's other slots retire first; use `BREAK` (emulator-only) for debugging stops.
2. `stage-control.md`: add `HALT` to §10.2 as `10.2.6` (syntax `HALT`, no operands, operation, example `HALT;;`), to the COND opcode list, to the §10.3 summary table, and rename the `BRANCH_COND_COUNT` row to a `COND_OP_COUNT` row covering the 6 non-NOP COND instructions (or add a row; keep existing parameter names if the RTL already uses them, see open question).
3. Requirements: IPU-ARCH-5 and IPU-DBG-8 say `HALT` instead of `END`; A-1 is marked done with a link to this issue. Add one sentence where `HALTED` is defined: the `HALT` instruction ends the kernel; it is not one of the causes that enter `HALTED`, unless the requirements say otherwise (open question).
4. Regenerate docs (E02-04 snapshots) so the instruction reference shows the new text.

## Tests

- [ ] E02-06 `test_bkpt_halts_after_whole_word_retires` passes unchanged (renamed by E03-19's script).
- [ ] `bazel build //docs:build_docs` has no new warnings.
- [ ] E02 baselines: only generated docs and `isa.json` text change.

## Acceptance Criteria

- [ ] ISA text, RTL control spec and requirements describe the same `HALT`: COND slot, opcode as `BKPT` today, ends the kernel after the word retires.
- [ ] A-1 is closed.

## Open questions

- After `HALT`, does the IPU go to `IDLE`/hand-off (kernel done, IPU-DBG-8) or to `HALTED`? The requirements imply kernel done; this issue writes that and asks the requirements owner to confirm.
- Does the RTL repo use `BRANCH_COND_COUNT` as a parameter name? If so, add `HALT` without renaming the parameter.
