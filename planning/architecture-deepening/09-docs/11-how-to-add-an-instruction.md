# How-to: add an instruction

Part of [Epic E09: Documentation restructure](00-epic.md) · ID E09-11 · Size S · Depends on: E09-02, E09-06, E03 ("one edit" CI test, strict schema, HARDWARE/EMULATOR/ASSEMBLER tags, derived syntax), E04 (per-slot handler modules written against format codecs) · Behaviour change: none

## Goal

A contributor adds an instruction by following one page, and the page shows
exactly the three edits the "one edit" test proves are enough: the ISA entry,
the handler, the test.

## Background

- `adding-instruction.md` (346 lines) describes the current flow: an
  `instruction_spec.py` entry, an `execute_*` method in `ipu.py`, a hand-built
  instruction dict in tests. E09-02 fixed its wrong facts but not its
  structure.
- Decision Q16: adding an instruction is the ISA package entry, an execute
  handler and a test; everything else is derived, and a CI test adds a dummy
  instruction to prove it. E03 builds that test.
- Q17, Q39, Q41, Q42: each slot and opcode is tagged HARDWARE, EMULATOR or
  ASSEMBLER. Only HARDWARE reaches the SystemVerilog package; EMULATOR
  instructions are rejected by `--target hardware`.
- Q15: `syntax` is derived from the mnemonic and operands, so it is no longer
  written by hand.
- Q45: mnemonic standard (control stage plain, datapath `UNIT.OP[.VARIANT]`, no
  underscores).
- Q48, Q67: handlers live in one module per slot, decode lanes through the
  format codec, compute, encode. No `if wide` branches.
- `CLAUDE.md` "Adding an Instruction" holds a five-step checklist.

## Implementation

1. Write `how-to/add-an-instruction.md` with `anthropic-skills:technical-writing`:
   - Before you start: choose the slot, the tag (HARDWARE, EMULATOR,
     ASSEMBLER) and the mnemonic under the standard. Link the ISA reference.
   - Step 1: the ISA entry in `ipu_isa`, using a real example included from
     the E03 "one edit" test fixture, so the example is tested.
   - Step 2: the handler in the slot's module, written against the format
     codec. Show the keyword-only signature rule.
   - Step 3: the test, with the shared test support from E04 (`conftest`).
   - Step 4: run `bazel test //...`; for a HARDWARE instruction, check the
     SystemVerilog package (link the `CLAUDE.md` section; widths are wire
     format).
   - Pseudo-instructions: when an assembler expansion is enough.
2. Delete `adding-instruction.md`; fix inbound links.
3. Make `CLAUDE.md` "Adding an Instruction" a short checklist that links this
   page (edit with `mattpocock-skills:writing-for-agents`). The checklist and the
   page must list the same steps.

*As built:* following the page on scratch edits (an `lr` HARDWARE instruction and an `acc_store` EMULATOR one) showed that the three edits are enough for the code but not for the tests. The tests that pin today's ISA fail. `tools/update_baselines.sh` regenerates nine baseline files. Four facts are pinned by hand and have no update target: `ipu-isa/test/fixtures/spec_snapshot.json`, the reserved-opcode counts in `test_faults.py`, the F-1 table in `requirements/faults.md`, and the one-edit tests' own count of `lr` instructions. The page lists them in step 4. A follow-up could derive or regenerate them.

## Tests

- [ ] `bazel test //docs:test_doc_asm_blocks //docs:test_doc_commands` passes.
- [ ] Follow the page on a scratch branch to add one HARDWARE and one EMULATOR
      instruction; `bazel test //...` and the SystemVerilog package test pass,
      and `--target hardware` rejects the EMULATOR one. Record it in the PR.

## Acceptance Criteria

- [ ] The page lists exactly the edits the "one edit" test requires, and no
      others.
- [ ] Code examples are included from tested files, not copied.
- [ ] `CLAUDE.md` and the page agree on the steps.
