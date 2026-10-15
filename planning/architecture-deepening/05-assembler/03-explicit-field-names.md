# Decoded-word field names from an explicit table

Part of [Epic: Assembler as a deep module](00-epic.md) · ID E05-03 · Size S · Depends on: E05-02, E03 (binary codec), E03-05 (operand-type registry), E02-02 (field-name snapshot) · Behaviour change: none

## Goal

The names of the bit fields in a decoded instruction word (`lr_inst_0_token_0_lr_inst_opcode` and so on) are part of the wire format: the emulator, the debugger, test fixtures and the Cycle Inspector traces all key on them. Today they come from Python class names. Make them an explicit, frozen table next to the codec, so renaming a class can no longer change them (L9).

## Background

- `CompoundInst.get_fields()` (`ipu_as/compound_inst.py:145-178`) builds names from `camel_case_to_snake_case(inst_type.__name__)` (`:158`) and `token_type.__name__` (`:173`).
- The emulator rebuilds the same strings with its own tables: `_TYPE_FIELD_SUFFIX` and `_SLOT_FIELD_PREFIX` in `ipu_emu/ipu.py`, plus `_opcode_field` and the LR opcode field lookup in the same file.
- `src/tools/ipu-apps/test/mobilevit/fixture_packed_l{4,5}_measure.py` hard-code the strings.
- `debug_pipeline.stage_operations` (`ipu_emu/debug_pipeline.py:15-24`) re-packs the dict using `get_fields()`.

## Implementation

1. In the `ipu_isa` codec module, coordinating with E03 (add it there if E03 has not), define:

   ```python
   @dataclass(frozen=True)
   class FieldSpec: name: str; slot: str; instance: int; index: int; bits: int; shift: int

   SLOT_FIELD_PREFIX: Mapping[str, str]       # "lr" -> "lr_inst"           (explicit, today's _SLOT_FIELD_PREFIX)
   TYPE_FIELD_SUFFIX: Mapping[str, str]       # operand type -> "lr_idx" etc. (explicit, today's _TYPE_FIELD_SUFFIX)
   WORD_LAYOUT: tuple[FieldSpec, ...]         # LSB-first; names = prefix + instance + token index + suffix
   def field_name(slot: str, instance: int, index: int) -> str
   def opcode_field(slot: str, instance: int = 0) -> str
   ```

   The two name tables are written-out data keyed by ISA names (slot names and operand-type names), never by Python class names. Per D2, a new operand type should need one edit. The suffix is therefore read from the `field_suffix` attribute of the E03-05 operand-type registry; `TYPE_FIELD_SUFFIX` is a view over it, not a second table. An import-time assertion checks that every operand type and slot has an entry.
2. `CompoundInst.get_fields()` returns `[(f.name, f.bits) for f in WORD_LAYOUT]`.
3. Replace the string building in `ipu_emu/ipu.py` (`_TYPE_FIELD_SUFFIX`, `_SLOT_FIELD_PREFIX`, `_opcode_field`, the LR opcode field names) with `field_name` and `opcode_field`. This is a small edit in E04's area; coordinate with E04.
4. Replace the literal strings in the two mobilevit fixtures with `field_name(...)` calls.

## Tests

- [ ] Snapshot: `WORD_LAYOUT` names and widths equal the E02 field-name snapshot.
- [ ] A test renames a token class at runtime (monkeypatch `__name__`) and asserts `WORD_LAYOUT` is unchanged.
- [ ] E02 digests, SV package golden and `isa.json` snapshot unchanged.
- [ ] `bazel test //...` green.

## Acceptance Criteria

- [ ] No field name on the decode path is derived from `__name__`. `grep camel_case_to_snake_case src/tools` finds no decode-path use.
- [ ] The emulator imports field names instead of building strings.

## Risks

- `gen_codegen.py:370` still derives SV enum typedef names from class names. That is the SV contract (CLAUDE.md: field names become SV field names). Leave it to E08-13, which moves `gen_codegen` and pins typedef names the same way. The E02 SV golden must stay identical.
