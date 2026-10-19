# Emulator reads the ISA through the typed view

Part of [Epic E03: ISA package](00-epic.md) · ID E03-10 · Size M · Depends on: E03-09 · Behaviour change: none · `*_WIDTH` localparams: unchanged

## Goal

Move every ISA lookup in `ipu_emu` from string-keyed dicts to `ipu_isa.isa()`. The emulator then depends on the ISA module's interface, not on the authoring format, and E04 can rebuild dispatch on typed records.

## Background

- `ipu_emu/ipu.py` imports `INSTRUCTION_SPEC`, `SLOT_BINARY_LAYOUT`, `SLOT_UNIONS` (`ipu.py:36-38`) and walks them in `_build_all_instruction_field_maps` (`:265-290`), `_INSTRUCTION_READ_TYPES` (`:293-305`), `_build_plan` (`:451-499`) and `_build_slot_plans` (`:502-530`).
- `_READ_CODES = {"LrIdx": ..., "CrIdx": ..., "LcrIdx": ...}` (`ipu.py:332`) keys on type-name strings; `_resolve_operand` (`:795-833`) branches on the same strings.
- `debug_pipeline.py:5` imports `COMPOUND_LAYOUT_SLOT_ORDER`, `SLOT_COUNT`, `SLOT_UNIONS`, `is_hardware_slot`.
- E03-05 and E03-06 already replaced `_TYPE_FIELD_SUFFIX` and `_SLOT_FIELD_PREFIX` with registry lookups.

## Implementation

1. `_build_slot_plans`, `_build_plan` and the field-map builders iterate `isa().instructions(slot)` and read `Instruction.operands`, `Operand.type`, `Operand.read`, `Instruction.execute_fn`.
2. `_READ_CODES` keys on `OperandType` (or `OperandType.name` read from the record), not on literals; `_resolve_operand` takes the `OperandType`.
3. `debug_pipeline` reads `SLOTS`, `LAYOUT_ORDER`, `SlotInfo.tag` and `isa().layout`.
4. Leave the plan structure, handler signatures and the cycle loop unchanged; E04 owns them.

## Tests

- [ ] E02 decode snapshot `emulator_field_maps.json` unchanged.
- [ ] E02 timing tests, kernel case records and `bazel test --define=ipu_proto=1 //src/tools/ipu-emu-py/...` unchanged.
- [ ] `grep -n "INSTRUCTION_SPEC\|SLOT_UNIONS\|SLOT_BINARY_LAYOUT" src/tools/ipu-emu-py/src` finds nothing.

## Acceptance Criteria

- [ ] `ipu_emu` reads ISA facts only through `ipu_isa.isa()` and the registries.
- [ ] No operand-type or slot name appears as a string literal in `ipu.py` dispatch code.
