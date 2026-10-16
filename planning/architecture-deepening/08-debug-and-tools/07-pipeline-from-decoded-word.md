# Pipeline view from the structured decoded word

Part of [Epic: Debugger, generators and editor](00-epic.md) · ID E08-07 · Size M · Depends on: E08-04, E03 (codec `DecodedWord` / `DecodedSlot` in `ipu_isa`) · Behaviour change: none

## Goal

The debugger's pipeline and disassembly views read slots, opcodes and operands from the structured decoded word that the `ipu_isa` codec returns. They stop re-packing the field dict into an integer, disassembling it to text and splitting that text on `;`. Slot labels come from the ISA slot metadata instead of a hard-coded list.

## Background

E03-18 already did step 2 and part of step 4: `stage_operations` calls `ipu_isa.codec.disasm.slot_operations`, `semantic_operations` and the `;` splitting in `pipeline.py` are gone, and `disassemble_at` uses `disassemble`. What is left: the TUI's `_disassembly` still decodes each word twice (session text, then `stage_operations`) and `_all_operations` / the `slot_labels` zip split the joined text again; use the labelled pairs directly. The bullets below describe the code before E03-18.

- `pipeline.stage_operations` (old `debug_pipeline.py:15-24`) re-packs the field dict with `CompoundInst.get_fields()`, calls `CompoundInst.decode(word)`, splits on `";"`, and re-applies operand bindings to the text (`semantic_operations`).
- `cli.disassemble_at` (old `debug_cli.py:361-378`) re-packs the dict the same way.
- `SLOT_LABELS` (old `debug_pipeline.py:59`) hard-codes `("CTRL", ("LR0", "LR1", "LR2", "COND", "BREAK"))`. A change to the LR sub-slot count, or a slot added in `instruction_spec.py`, does not reach the view (D2).

## Implementation

1. Use the codec's word decode, assumed `DecodedWord.from_fields(fields)` / `.slots() -> tuple[DecodedSlot, ...]`, where `DecodedSlot` has `slot`, `instance`, `label`, `opcode`, `operands` and `text()` (E03; adjust to the names E03 lands).
2. `stage_operations(fields)` becomes `[(s.label, s.semantic_text()) for s in DecodedWord.from_fields(fields).slots()]`. Delete `semantic_operations` and the text splitting.
3. Build `SLOT_LABELS` and the stage grouping from the ISA slot metadata (slot order and stage from the E03 timing contract; instance counts from the codec).
4. `DebugSession.disassemble` uses `DecodedWord.disassemble()`.

## Tests

- [ ] For every program in the E02 digest set, the pipeline view text at every PC equals the base-commit output (snapshot recorded first).
- [ ] A synthetic ISA with a fourth LR sub-slot shows `LR3` in the view with no debugger edit (if E03's view allows a synthetic ISA in tests).
- [ ] `bazel test //...` green.

## Acceptance Criteria

- [ ] `grep -n "split(\";\")\|get_fields" src/tools/ipu-debug` is empty.
- [ ] No slot name is hard-coded in `ipu_debug`.
