# Derived layout computed once, immutable, no import-time `setattr`

Part of [Epic E03: ISA package](00-epic.md) · ID E03-08 · Size S · Depends on: E03-05 · Behaviour change: none · `*_WIDTH` localparams: unchanged (same solver, same inputs)

## Goal

The union layout (field packing per slot and the widths it derives) is a pure function of the spec. Compute it once into an immutable value and let every consumer ask for it. Remove the import-time mutation of other modules, so correctness no longer depends on which module is imported first.

## Background

- `SLOT_BINARY_LAYOUT` and `SLOT_UNIONS` start as empty placeholders (`instruction_spec.py:140-141`) and are filled by `.update()` at the bottom of the module (`:1789-1797`).
- `finalize_derived_operand_bits()` (`union_layout.py:268-280`) `setattr`s `LR_INC_DEC_IMM_FIELD_BITS` on `lr_inc_dec_imm` and `RESHAPE_MASK_FIELD_BITS` on `reshape_mask`, both initialised to `0` (`lr_inc_dec_imm.py:12`, `reshape_mask.py:21`). Any module that does `from ... import LR_INC_DEC_IMM_FIELD_BITS` before `instruction_spec` is imported binds `0`. The accessor functions raise `RuntimeError("... union layout must be computed first")` to paper over this.
- `SlotUnion` and `UnionField` are mutable dataclasses (`union_layout.py:95+`); `_pad_slot_to_target` mutates `field.bits` in place (`:247-265`).

## Implementation

1. Add `ipu_isa/layout.py`:
   ```python
   @dataclass(frozen=True)
   class IsaLayout:
       unions: Mapping[Slot, SlotUnion]          # SlotUnion / UnionField frozen; tuples, MappingProxyType
       operand_bits: Mapping[str, int]           # every operand type, derived widths filled in
       def slot_width(self, slot: Slot) -> int: ...
       def word_bits(self, tag: Tag | None = None) -> int: ...   # from E03-07

   def compute_layout(spec, operand_types, slots) -> IsaLayout    # pure; used by tests and E03-24
   @functools.cache
   def isa_layout() -> IsaLayout                                  # the default spec
   ```
   The solver builds new frozen objects instead of mutating; padding returns a new field.
2. `SLOT_UNIONS` and `SLOT_BINARY_LAYOUT` are assigned once from `isa_layout()` (no placeholder, no `.update`).
3. Delete `finalize_derived_operand_bits` and `_DERIVED_OPERAND_TARGETS`. In `lr_inc_dec_imm.py` and `reshape_mask.py` replace the module constants with functions `lr_inc_dec_imm_bits()` / `reshape_mask_field_bits()` reading `isa_layout().operand_bits`. Update the callers (`ipu_as/immediate.py:9-21`, the emulator) in the same PR; no PEP 562 compatibility shim (D5).
4. `layout.py` imports the spec modules, not the other way round, so there is no import cycle and no lazy import.

## Tests

- [ ] Fresh-interpreter test: import `ipu_as.immediate` before anything else and assert the derived widths are non-zero (regression for the ordering hazard).
- [ ] `compute_layout` called twice returns equal values; mutating a returned mapping raises `TypeError`.
- [ ] Existing `test_union_layout.py` passes against the frozen types.
- [ ] E02 baselines unchanged (SV golden, decode snapshots, asm digests).

## Acceptance Criteria

- [ ] No module in `ipu_isa` assigns attributes on another module.
- [ ] No placeholder global is filled after definition.
- [ ] `compute_layout` is a pure function that E03-24's "one edit" test can call with a modified spec.
