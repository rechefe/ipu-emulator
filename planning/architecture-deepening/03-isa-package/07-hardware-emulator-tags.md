# HARDWARE / EMULATOR / ASSEMBLER tags and the hardware word

Part of [Epic E03: ISA package](00-epic.md) · ID E03-07 · Size S · Depends on: E03-06 · Behaviour change: none · `*_WIDTH` localparams: unchanged (this issue declares data only; E03-22 applies it to the SV package)

## Goal

Every slot, instruction and pseudo-instruction carries one tag (Q17): `HARDWARE` (real IPU), `EMULATOR` (exists only in the emulator) or `ASSEMBLER` (expanded by the assembler, no opcode). From the tags, `ipu_isa` derives the hardware word: the hardware slots in layout order, with the emulator-only slots after them (Q41). SV, RTL docs (E03-22, E03-23) and the assembler's `--target hardware` (E05-15) consume this view.

## Background

- `SLOT_METADATA["acc_store"]` has `"hardware": False` (`instruction_spec.py:176`); `is_hardware_slot` (`:186-188`) reads it. The comment at `:171-172` says such slots are "excluded from HW codegen", but `gen_codegen.py` never calls `is_hardware_slot`: the vendored package contains `acc_store_slot_t` and `IPU_COMPOUND_INST_WIDTH = 186`. The only caller is `debug_pipeline.py:6-11` (it adds `*` to the label).
- The break slot (`BREAK`, `BREAK.IFEQ`) has no flag; decision Q39 tags it EMULATOR. D6: `acc_store` is not real hardware.
- `COMPOUND_LAYOUT_SLOT_ORDER` (`instruction_spec.py:159-169`) is `cond, lr, load, mult, acc, aaq, store, acc_store, break`, MSB to LSB. The two emulator-only slots are already the 31 least significant bits (ACC_STORE 9, BREAK 22), so the hardware word is the top 155 bits of the 186-bit emulator word. Requirement OQ-14 (`docs/content/requirements/open-questions.md:37`) proposes the same: the emulator word without the ACC_STORE and BREAK slots.
- Q40 makes `BKPT` (cond slot) the hardware instruction `HALT` (rename in E03-19, RTL docs in E03-20). It is tagged HARDWARE here because the cond slot is hardware.
- Pseudo-instructions `B`, `BZ`, `BNZ`, `BGT`, `BLE` (`PSEUDO_INSTRUCTION_SPEC`) have no opcode.

## Implementation

1. Add `class Tag(StrEnum): HARDWARE; EMULATOR; ASSEMBLER` in `ipu_isa/tags.py`.
2. `SlotInfo.tag` replaces the `hardware` key: `acc_store` and `break` are `EMULATOR`; every other slot `HARDWARE`. Delete `is_hardware_slot`; `debug_pipeline` reads `SLOTS[slot].tag`.
3. Instructions take their slot's tag by default. Schema rules:
   - an instruction in an EMULATOR slot is EMULATOR;
   - every pseudo-instruction is ASSEMBLER and expands only to instructions tagged HARDWARE or EMULATOR;
   - in `LAYOUT_ORDER`, no HARDWARE slot follows an EMULATOR slot (Q41).
4. Derived view in `ipu_isa/layout.py` (pure functions of the registries and union layout):
   ```python
   def slots_with_tag(tag: Tag) -> tuple[Slot, ...]          # in LAYOUT_ORDER
   def word_bits(tag: Tag | None = None) -> int               # None: emulator word (186); HARDWARE: 155
   ```
5. Correct the comment at `instruction_spec.py:171-172` to say what is true until E03-22: the SV package still includes emulator-only slots.

## Tests

- [ ] `ipu-isa/test/test_tags.py`: `word_bits() == 186`, `word_bits(Tag.HARDWARE) == 155`; hardware slots are a prefix of `LAYOUT_ORDER`; each schema rule fails on a planted violation (EMULATOR slot placed before `store`; pseudo tagged HARDWARE).
- [ ] E02 baselines unchanged (SV golden included: no generator reads the tag yet).

## Acceptance Criteria

- [ ] Every slot, instruction and pseudo-instruction has exactly one tag.
- [ ] The hardware word width is derived, not written anywhere.

## Open question

- May an EMULATOR-tagged instruction live inside a HARDWARE slot (it would decode as a reserved encoding, F-1, on hardware)? No such instruction exists today; until decided, the schema rejects it.
