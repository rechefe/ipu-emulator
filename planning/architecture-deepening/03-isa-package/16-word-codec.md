# Instruction-word codec in `ipu_isa`: field layout, decode, encode, program bytes

Part of [Epic E03: ISA package](00-epic.md) · ID E03-16 · Size M · Depends on: E03-05, E03-06, E03-07, E03-08 · Behaviour change: none · `*_WIDTH` localparams: unchanged

## Goal

`ipu_isa` owns the binary format of an instruction word (Q69): the field layout with explicit field names, decoding a word into fields, encoding fields into a word, and the byte layout of a program file. The emulator decodes programs through it and stops using the assembler for that. The field names are an explicit table built from the registries, so renaming a Python class can never change them.

## Background

- `execute.decode_instruction_word` and `load_binary_instructions` (`ipu_emu/execute.py:42-87`) call `ipu_as.compound_inst.CompoundInst.get_fields()` and `.bits()` (`execute.py:30`). `_instruction_aligned_bytes` (`execute.py:61-71`) restates `ipu_as.lark_tree.instruction_aligned_bytes_len()` (`lark_tree.py:206-210`): 32-bit aligned, little-endian.
- `get_fields()` (`compound_inst.py:145-178`) derives names from class names: `<snake(InstClass)>[_<i>]_token_<k>_<snake(TokenClass)>`. E03-05 and E03-06 already hold the same strings explicitly (`OperandType.field_suffix`, `SlotInfo.field_prefix`, `SlotInfo.opcode_enum`).
- `debug_cli.disassemble_at` (`debug_cli.py:361-380`) and `debug_pipeline.stage_operations` (`debug_pipeline.py:15-25`) re-encode a decoded dict into a word by hand.
- E05-02 and E05-03 assume this codec exists in `ipu_isa` (`SlotCodec`, `WORD_LAYOUT`, `field_name`, `opcode_field`). E05-15 (`--target hardware`) needs the hardware part of a word.

## Implementation

1. `ipu_isa/codec/word.py`:
   ```python
   @dataclass(frozen=True, slots=True)
   class FieldSpec:
       name: str          # "lr_inst_0_token_0_lr_inst_opcode" — wire format
       slot: Slot
       instance: int      # 0..count-1
       index: int         # 0 = opcode, k = union field k-1
       bits: int
       shift: int         # LSB position in the emulator word

   WORD_LAYOUT: tuple[FieldSpec, ...]                 # LSB first, built from isa_layout() and the registries
   def field_name(slot: Slot, instance: int, index: int) -> str
   def opcode_field(slot: Slot, instance: int = 0) -> str

   class DecodedWord(dict[str, int]):                 # same keys as today, so emulator dispatch `inst[key]` is unchanged
       @classmethod
       def from_word(cls, word: int) -> "DecodedWord"
       def to_word(self) -> int
       def slot_value(self, slot: Slot, instance: int = 0) -> int

   def word_bytes(tag: Tag | None = None) -> int      # 32-bit aligned: 24 for the emulator word, 20 for the hardware word
   def hardware_word(word: int) -> int                # the top word_bits(HARDWARE) bits (E03-07 ordering)
   def unpack_program(data: bytes) -> list[DecodedWord]
   def pack_program(words: Sequence[int]) -> bytes
   ```
2. `WORD_LAYOUT` is built from the layout and registries, never from class names. An import-time check in `ipu_as` asserts `CompoundInst.get_fields()` equals `[(f.name, f.bits) for f in WORD_LAYOUT]` until E05-03 makes `get_fields()` return it.
3. `ipu_emu/execute.py`: `decode_instruction_word` becomes `DecodedWord.from_word`; `load_binary_instructions` becomes `unpack_program`; delete `_instruction_aligned_bytes`. `debug_cli` and `debug_pipeline` use `DecodedWord.to_word()` instead of hand re-encoding.
4. `ipu_as.lark_tree.instruction_aligned_bytes_len` and the `.bin` writer call `word_bytes()` and `pack_program()`.

## Tests

- [ ] `ipu-isa/test/test_word_codec.py`: `WORD_LAYOUT` equals E02-02's `word_fields.json`; `from_word(w).to_word() == w` for every word of the E02-01 corpus; `word_bytes() == 24`, `word_bytes(Tag.HARDWARE) == 20`; `hardware_word` drops exactly the ACC_STORE and BREAK fields.
- [ ] `unpack_program(pack_program(words))` round trip; output equals today's `.bin` bytes for the whole corpus (E02-01 digests).
- [ ] E02 decode snapshots, disassembly and kernel case records unchanged.

## Acceptance Criteria

- [ ] Word layout, field names and program byte layout are defined once, in `ipu_isa`.
- [ ] `ipu_emu/execute.py` imports nothing from `ipu_as`.
- [ ] E05-02/E05-03 can use this codec without adding to it.

## Risks

- `DecodedWord` subclasses `dict` to keep about 60 test sites and the hot dispatch path unchanged; a typed object is a later decision (E04 plans may compile it away).
