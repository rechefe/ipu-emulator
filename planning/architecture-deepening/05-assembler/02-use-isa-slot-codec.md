# Encode through the `ipu_isa` slot codec; delete the nine `Inst` subclasses

Part of [Epic: Assembler as a deep module](00-epic.md) · ID E05-02 · Size M · Depends on: E05-01, E03 (binary codec in `ipu_isa`), E03-06 (slot registry with `doc_intro`) · Behaviour change: none

## Goal

The assembler stops owning instruction encoding. It turns parsed text into operand values and calls the slot codec that E03 places in `ipu_isa` (Q69). The nine copy-paste `Inst` subclasses go away. What is left in `ipu_as` is text handling: tokens, operand parsing, slot assignment within a word, labels and error locations.

This deepens the module. The interface (`assemble(text) -> words`) stays the same, while a large amount of duplicated code is removed.

## Background

- `inst.py:365-686` defines `LoadInst`, `StoreInst`, `AccStoreInst`, `MultInst`, `AccInst`, `AaqInst`, `LrInst`, `CondInst` and `BreakInst`. Each is about 35 lines, and they differ only in the slot string, the opcode class, `nop_inst`, and the doc heading and intro.
- `CompoundInst._fill_instructions` (`compound_inst.py:26-81`) finds the slot by scanning subclasses (`Inst.find_inst_type_by_opcode`). It places instructions into slot instances (three LR sub-slots), and `_fill_out_nop` (`compound_inst.py:83-87`) pads with NOPs.
- Generators also use the classes:
  - `gen_docs` uses `Inst.get_all_instruction_classes()` and `.description()`;
  - `gen_codegen` uses `CompoundInst.instruction_types()` and `_slot_type_name()`;
  - `compound_inst.generate_struct_layout_svg` (`compound_inst.py:200`) uses the class legend.

  They move to `ipu_gen` in E08 and should read the codec or ISA view directly.
- Q69: the binary codec (encode and decode of one slot and of a whole word) lives in `ipu_isa`.

## Implementation

1. The codec interface E03-16 and E03-17 provide:

   ```python
   # ipu_isa.codec.slot (E03-17)
   class SlotCodec:
       slot: Slot; instances: int; bits: int
       def encode(self, mnemonic: str, operands: Sequence[int]) -> int  # encoded values, assembly order; raises EncodeError
       def nop(self) -> int
       def decode(self, value: int) -> DecodedSlot                      # raises ReservedEncoding for an unused opcode
       def instruction(self, mnemonic: str) -> Instruction
   SLOT_CODECS: Mapping[Slot, SlotCodec]
   OPCODE_SLOT: Mapping[str, Slot]         # upper-case mnemonic -> slot; NOP is absent (it is in every slot)
   WORD_CODEC: WordCodec                   # .pack(Mapping[tuple[Slot, int], int]) -> int (missing = NOP)
                                           # .unpack(word) -> dict[tuple[Slot, int], DecodedSlot]
   def codecs(view: Isa | None = None) -> Codecs   # .slots, .word, .opcode_slot for a modified ISA
   # ipu_isa.codec.word (E03-16): WORD_LAYOUT, FieldSpec, field_name, opcode_field,
   # DecodedWord, word_bytes, hardware_word, pack_program, unpack_program, unpack_words
   ```

   `Slot` is a `str` Enum, so a slot name such as `"lr"` finds the same entries.

2. In `ipu_as`, add `encode.py`:
   - `operand_value(token: AnnotatedToken, operand_type, symbols) -> int` validates and converts one operand. Range checks and enum lookups use the operand-type metadata from `ipu_isa`, and errors carry the token's location.
   - `encode_word(parsed: ParsedCompound, symbols) -> int` assigns slots (the same rule as `_fill_instructions`: the first free instance per slot, and `nop` fills any slot), converts operands, calls `SLOT_CODECS[slot].encode` and `WordCodec.pack`.
3. Re-express `CompoundInst.encode` and `CompoundInst.decode` as thin calls to `encode_word` and the codec's word decode. Keep the `CompoundInst` public methods that tests and generators call, delegating to the codec, until E08 moves the generators.
4. Delete the nine subclasses and `_build_struct_table`. The per-slot doc headings and intros come from the E03-06 slot registry (`doc_intro`).
5. The `ipu_token` classes keep parsing and validating text. Their `encode()` methods and bit-width knowledge are removed where the codec now owns them. Keep `decode` text formatting only if E03's `DecodedSlot` does not cover disassembly text.

## Tests

- [ ] E02 `.asm` digest test unchanged. The disassembly text snapshot (E02) is unchanged too.
- [ ] Round trip for every slot and opcode in `INSTRUCTION_SPEC`: assemble a one-instruction word, decode it, and the disassembled text re-assembles to the same word.
- [ ] Error parity: wrong operand count, out-of-range immediate, unknown mnemonic and too many instructions for a slot each still fail. The message text is unchanged, checked against `test_diagnostics` expectations.
- [ ] `bazel test //...` green; `test_sv_package_compiles` and `test_gen_codegen` unchanged.

## Acceptance Criteria

- [ ] `inst.py` has no per-slot subclasses. `grep -n "class .*Inst(Inst)" ipu_as` is empty.
- [ ] Encoding and decoding bit layouts are computed only in `ipu_isa`.
- [ ] The encode stage is at least 3x faster than at the base commit, combined with E05-01 (Q71). The tracked `bench_assemble` numbers are in the PR.
- [ ] Generated SV package, `isa.json` and docs are byte-identical (E02 snapshots).

## Risks

- Slot errors embed the class name in user-visible text: `compound_inst.py:66` produces "Instruction type LrInst is not allowed …" and `:71` produces "Too many instructions of type LrInst …". The editor keys on message text (`editing.js:179`). Keep the text identical with a `LEGACY_SLOT_NAME = {"lr": "LrInst", …}` table in the error builder. Any wording change is a separate behaviour-change PR.
- The simulation-only slot warning (`compound_inst.py:74-79`, `is_hardware_slot`) must survive the move. E05-15 builds on it.
- Generators depend on class names (`gen_codegen.py:370` enum typedef names come from class names). Keep a compatibility shim that exposes the same names until E05-03 and E08 replace them with explicit tables. The E02 SV-package golden catches any drift.
