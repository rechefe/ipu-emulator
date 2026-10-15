# SV package generated from the HARDWARE set only

Part of [Epic E03: ISA package](00-epic.md) · ID E03-22 · Size S · Depends on: E03-07, E03-11 · Behaviour change: **yes, wire format** · `*_WIDTH` localparams: **CHANGE (breaking)**: `IPU_COMPOUND_INST_WIDTH` 186 → 155; `ACC_STORE_SLOT_U_WIDTH`, `ACC_STORE_SLOT_T_WIDTH`, `BREAK_SLOT_U_WIDTH`, `BREAK_SLOT_T_WIDTH` removed; every other `*_WIDTH` unchanged

## Goal

Decisions D6, Q17, Q39 and Q41: the SystemVerilog package the RTL repo vendors describes only what hardware decodes. The emulator-only slots (ACC_STORE, BREAK) and their types leave the package. The emulator binary keeps all 186 bits (Q41), and because emulator-only slots are the least significant bits (E03-07), the hardware word is the top 155 bits of the emulator word. This is a breaking change for the downstream repo and must be announced as one (CLAUDE.md).

## Background

- `gen_codegen.py` renders every slot: `_slot_union_descriptors` and `_compound_members` iterate all slots (`gen_codegen.py:240-334`); `is_hardware_slot` is never called. The package therefore contains `acc_store_inst_opcode_t`, `break_inst_opcode_t`, `acc_store_slot_u/t`, `break_slot_u/t`, and `ipu_compound_inst_t` with `acc_store_slot` and `break_slot` members, total 186 bits.
- Requirements already expect this: IPU-ARCH-4 "The IPU shall implement every ISA slot except `ACC_STORE` and `BREAK`" (`requirements/architecture.md:86`); OQ-14 proposes "the emulator word without the `ACC_STORE` and `BREAK` slots, padded to a multiple of 32 bits" (`requirements/open-questions.md:37`).
- Widths: ACC_STORE slot 9 bits, BREAK slot 22 bits; 186 − 9 − 22 = 155. Padded to 32-bit beats: 160 bits (20 bytes, `word_bytes(Tag.HARDWARE)`, E03-16).
- `test_gen_codegen.py:32-58` asserts `acc_store_slot_t` is present and `IPU_COMPOUND_INST_WIDTH = {CompoundInst.bits()}`; `:24-29` asserts the last compound member starts with `break_slot`.

## Implementation

1. `gen_codegen` takes the slot set from `slots_with_tag(Tag.HARDWARE)` (E03-07) and the instructions from the typed view filtered by tag. Enum descriptors are emitted only for types used by a hardware slot.
2. `compound_width` in the render context is `word_bits(Tag.HARDWARE)`.
3. The compound comment states that the struct is the hardware word and that the emulator word appends the ACC_STORE and BREAK slots below bit 0 of it (one sentence, generated from `LAYOUT_ORDER`).
4. Update tests: `test_gen_codegen.py` (no `acc_store_slot_t`; last member `store_slot`; width equals `word_bits(Tag.HARDWARE)`), `test_sv_package_compiles.py` (unchanged elaboration check), and the E02-03 localparam dict: edit `IPU_COMPOUND_INST_WIDTH` to 155 and delete the four removed names, as a visible diff in this PR.
5. Requirements: mark OQ-14's width part answered (155 bits, 160 padded) with a link to the derived value; the AXI packing part stays open.
6. Announce in the PR description and the release log: the localparam diff (`tools/sv_localparam_diff.sh <base>` output), the removed typedefs and members, and the statement that `.bin` files and the emulator are unchanged.

## Tests

- [ ] `test_localparams_match_table` passes with the edited dict; `tools/sv_localparam_diff.sh <base>` shows exactly the five changes in the header line.
- [ ] `test_sv_package_compiles` passes (slang, zero diagnostics).
- [ ] A test asserts no EMULATOR-tagged slot, instruction or enum used only by one appears in the rendered text.
- [ ] E02-01 asm digests and E02-02 field names unchanged (the binary format is untouched).

## Acceptance Criteria

- [ ] The SV package contains only HARDWARE-tagged slots and instructions.
- [ ] `IPU_COMPOUND_INST_WIDTH` is 155, derived, and every other remaining `*_WIDTH` is identical to the base commit.
- [ ] The breaking change is called out in the PR title and description with the localparam diff.

## Risks

- The downstream RTL repo pins a commit (Q46); it must adopt this deliberately. Land it together with, or right after, E03-19's rename so the downstream repo takes one breaking update, not two.
