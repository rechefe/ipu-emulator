# RTL-facing docs show the hardware word; emulator docs show both

Part of [Epic E03: ISA package](00-epic.md) · ID E03-23 · Size S · Depends on: E03-22 · Behaviour change: none for programs; generated docs change · `*_WIDTH` localparams: unchanged (SV handled in E03-22)

## Goal

Q17: SV and RTL docs come from the HARDWARE set only. After E03-22 the SV package describes the 155-bit hardware word, but the generated "Instruction format" page still shows the 186-bit emulator word as "the" instruction layout and links the SV package beside it. Make the page show the hardware word as the primary layout, show the emulator word as the hardware word plus the emulator-only slots, and mark EMULATOR and ASSEMBLER items in the instruction reference.

## Background

- `gen_docs.generate_instruction_format_md` (`ipu_as/gen_docs.py:492-535`) titles the section "Compound instruction layout ({width} bits)" with `width = CompoundInst.bits()` (186) and draws `generate_struct_layout_svg` and `generate_union_layout_svg` over all slots.
- The instruction reference marks ACC_STORE with a dagger "simulation-only" in one list (`gen_docs.py:166`) but not BREAK, and not per instruction.
- GLOSSARY.md defines *hardware word* and *emulator word*.

## Implementation

1. The instruction-format page: section "Hardware word (155 bits)" with the struct and union diagrams for HARDWARE slots; section "Emulator word (186 bits)" stating that the emulator appends ACC_STORE and BREAK below the hardware word, with their diagrams. Both widths come from `word_bits()` (E03-07).
2. The instruction reference: a tag column or badge per instruction (HARDWARE, EMULATOR, ASSEMBLER), generated from the typed view, replacing the hand-placed dagger.
3. The SVG renderers take the slot list as an argument (they already take `slot_order`); pass the tagged subsets.
4. Regenerate the E02-04 snapshots in this PR.

## Tests

- [ ] E02-04 snapshot diff limited to `instruction-format.md`, `instructions.md`, `programmer-guide.md` and the union SVG.
- [ ] A test asserts the page states both widths and that they equal `word_bits(Tag.HARDWARE)` and `word_bits()`.

## Acceptance Criteria

- [ ] An RTL reader sees the 155-bit hardware word first, matching the SV package.
- [ ] Every instruction's tag is visible in the reference.
