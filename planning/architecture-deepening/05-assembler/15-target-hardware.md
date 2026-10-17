# `--target hardware`: reject EMULATOR instructions, emit hardware words

Part of [Epic: Assembler as a deep module](00-epic.md) · ID E05-15 · Size M · Depends on: E05-02, E03-07 (HARDWARE / EMULATOR / ASSEMBLER tags and the derived hardware word) · Behaviour change: yes (new option; the default target `emulator` is unchanged)

## Goal

The same source can be assembled for the RTL (Q42):

- `ipu-as assemble --target hardware` rejects every instruction tagged EMULATOR, with a located error;
- it emits words containing only the hardware slots, which are the top bits of the emulator word (Q41);
- the default `--target emulator` emits today's full word, byte-identical.

## Background

- E03 tags every slot and opcode HARDWARE / EMULATOR / ASSEMBLER (Q17). The `break` slot (`BREAK`, `BREAK.IFEQ`) is EMULATOR (Q39), and `acc_store` is not real hardware (D6). `BKPT` becomes the hardware `HALT` (Q40).
- Q41: emulator-only slots stay in the emulator binary (186 bits) and are ordered after the hardware slots, so the hardware word is a prefix (the top bits) of the emulator word.
- Today the assembler only warns: `compound_inst.py:74-79` calls `is_hardware_slot` (`ipu_common/instruction_spec.py:186-188`) and emits `warnings.warn("... uses the simulation-only ... slot")`.
- Pseudo-instructions (ASSEMBLER tag) expand to real instructions (`lark_tree.py:49-64`), so their expansion's tags decide.

## Implementation

1. `AssemblySession(..., target: Literal["emulator", "hardware"] = "emulator")`.
2. Encoding with `target="hardware"`:
   - Any instruction whose opcode or slot is tagged EMULATOR raises `SlotError("BREAK is emulator-only and cannot be assembled for --target hardware")`, at the mnemonic's location. Implicit NOP fill of EMULATOR slots is not an error.
   - The word is `emulator_word >> EMULATOR_ONLY_BITS`, using the widths from the `ipu_isa` codec (`WordCodec.hardware_bits`). An import-time assertion checks that the emulator-only fields are exactly the low bits, the Q41 ordering. If E03's ordering puts them elsewhere, use the codec's `hardware_word(word)` instead of a shift.
   - `AssemblyResult.to_bin()` uses the hardware word's aligned byte length; `to_mem()` uses its hex width.
3. With `target="emulator"`, the simulation-only warning stays as today.
4. CLI: `--target [emulator|hardware]` on `assemble` and `check`. `check --target hardware` reports EMULATOR instructions as errors, so an RTL author sees them in the editor. Bazel: `assemble_asm(..., target = "emulator")`.
5. `disassemble --target hardware` reads hardware-width words.

## Tests

- [ ] For every E02 source that uses no EMULATOR instruction, the hardware words equal the top bits of the emulator words.
- [ ] A source with `BREAK` fails with `--target hardware` at the `BREAK` location, and assembles with the default target.
- [ ] Hardware `.bin` length = words × aligned hardware word bytes.
- [ ] Round trip: hardware-assemble, disassemble with `--target hardware`, re-assemble gives identical words.
  *As built:* the round trip needs source the assembler reads, so `disassemble --syntax assembly` (`ipu_isa.codec.disasm.program_text`) writes it; the default field listing is unchanged. `//src/tools/ipu-apps:test_assembly_round_trip` runs it over the whole kernel corpus for both targets, and `test_target.py` through the CLI.
- [ ] The default target digests are unchanged. `bazel test //...` green.

## Acceptance Criteria

- [ ] `--target hardware` exists on the CLI, the session and Bazel. Its word width is derived from `ipu_isa`, with no constant in `ipu_as`.
- [ ] The width of the hardware word equals the SV package's instruction width (cross-check against the E02 SV golden, `localparam` for the word width).

## Open question

- Should a hardware-target `.bin` carry a header or a different extension, so that loading it into the emulator by mistake fails loudly? Today's `.bin` has no header. The default here is no header and the same extension, and the emulator's loader checks the file length is a multiple of its word size.
