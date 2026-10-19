# Assembler, generators and fixtures read the typed view; dict helpers deleted

Part of [Epic E03: ISA package](00-epic.md) · ID E03-11 · Size M · Depends on: E03-09 · Behaviour change: none · `*_WIDTH` localparams: unchanged (E02-03 golden proves it)

## Goal

Finish the migration started in E03-10: every remaining consumer outside `ipu_isa` reads `ipu_isa.isa()`. Then delete the dict-returning helpers and make the authoring dicts private to `ipu_isa`, so the typed view is the only interface.

## Background

String-keyed reads of `INSTRUCTION_SPEC`, `PSEUDO_INSTRUCTION_SPEC`, `SLOT_UNIONS`, `SLOT_BINARY_LAYOUT` and the helper functions remain in:

- `ipu_as/inst.py` (8), `lark_tree.py` (7, pseudo-instruction expansion), `gen_vscode.py` (6), `gen_codegen.py` (5), `gen_docs.py` (3), `compound_inst.py` (2), `opcodes.py` (2);
- `ipu-apps/test/mobilevit/fixture_packed_l4_measure.py` and `fixture_packed_l5_measure.py` (3 each);
- tests: `ipu-isa/test/test_union_layout.py` (14), `ipu-as-py/test/test_gen_codegen.py` (3), `ipu_emu/isa_alias_spec` test (4).

## Implementation

1. Migrate each file above to `isa()` records and registries. One commit per file to keep review local.
2. `lark_tree` pseudo expansion uses `isa().pseudo(mnemonic, arity)` and `PseudoInstruction.expands_to`.
3. `gen_codegen`, `gen_docs` and `gen_vscode` take an `Isa` argument defaulting to `isa()`. This is what lets E03-24 render a modified ISA in-process.
4. Delete the dict-shaped helpers `get_instruction`, `get_instruction_by_opcode`, `get_operand_names_and_types`, `find_pseudo_instruction` and `get_opcode_for_instruction` from `ipu_isa/view.py`, where E03-09 moved them (`extract_opcodes` and `canonical_instruction_name` had no callers and are already gone). Make `instruction_tag` and `pseudo_tag` private to the validator.
5. Add a check to E03-02's purity test: outside `src/tools/ipu-isa/`, no source file imports `INSTRUCTION_SPEC`, `PSEUDO_INSTRUCTION_SPEC`, `SLOT_UNIONS` or `SLOT_BINARY_LAYOUT`.

## Tests

- [ ] All E02 baselines unchanged: asm digests, decode and disassembly snapshots, SV golden, generated artifacts.
- [ ] `test_diagnostics.py`, `test_hover_data.py`, `test_isa_data.py` unchanged.
- [ ] The import check fails on a planted `from ipu_isa.instruction_spec import INSTRUCTION_SPEC` in `ipu_as`.

## Acceptance Criteria

- [ ] No consumer outside `ipu_isa` touches the authoring dicts.
- [ ] Generators accept an `Isa` argument.
