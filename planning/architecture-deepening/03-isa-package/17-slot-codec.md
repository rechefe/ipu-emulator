# Slot codec in `ipu_isa`: encode and decode one instruction

Part of [Epic E03: ISA package](00-epic.md) · ID E03-17 · Size M · Depends on: E03-09, E03-16 · Behaviour change: none · `*_WIDTH` localparams: unchanged

## Goal

Complete the binary codec (Q69): turn one instruction (mnemonic plus operand values) into the bits of its slot and back, from the union layout alone. The assembler keeps only text handling and calls this codec (E05-02); the emulator and debugger decode slots through it; the "one edit" test (E03-24) can encode a new instruction with no assembler change.

## Background

- Encoding lives in nine copy-paste `Inst` subclasses (`ipu_as/inst.py:364-686`). Each `struct_by_opcode_table()` rebuilds the per-slot table on every call (`_build_struct_table`, `inst.py:54`), from `INSTRUCTION_SPEC` and `SLOT_UNIONS`. Operand bit placement follows `SlotUnion.opcode_bindings` (`union_layout.py`).
- Operand values are produced by token classes (`ipu_as/ipu_token.py`, `reg.py`, `immediate.py`): enum tokens map names to indices; numeric tokens check ranges.
- E05-02 assumes this interface (adjustable to what lands): `SlotCodec.encode(opcode, operands) -> int`, `nop()`, `decode(value) -> DecodedSlot`, `SLOT_CODECS`, `OPCODE_SLOT`, `WordCodec.pack(...)`.

## Implementation

1. `ipu_isa/codec/slot.py`:
   ```python
   @dataclass(frozen=True, slots=True)
   class DecodedSlot:
       slot: Slot
       instruction: Instruction          # from the typed view (E03-09)
       operands: tuple[int, ...]         # raw encoded values, in assembly operand order
       unused_bits: int                  # padding bits that were non-zero (F-1 evidence for E04-19)

   class SlotCodec:
       slot: Slot; instances: int; bits: int
       def encode(self, mnemonic: str, operands: Sequence[int]) -> int      # values already resolved to integers
       def decode(self, value: int) -> DecodedSlot                           # raises ReservedEncoding for an unused opcode
       def nop(self) -> int

   SLOT_CODECS: Mapping[Slot, SlotCodec]
   OPCODE_SLOT: Mapping[str, Slot]        # upper-case mnemonic -> slot, from the view

   class WordCodec:
       def pack(self, slot_values: Mapping[tuple[Slot, int], int]) -> int   # (slot, instance) -> slot bits; missing = NOP
       def unpack(self, word: int) -> dict[tuple[Slot, int], DecodedSlot]
   ```
   Tables are built once per `Isa` (cached), so encoding is a dict lookup plus shifts. `encode` checks operand count and each value's range from `OperandType.domain` and raises `EncodeError(IpuError, ValueError)` naming the operand; text locations are added by the assembler (E05-04).
2. `codec` takes an `Isa` argument defaulting to `isa()`, so E03-24 can encode with a modified ISA.
3. Cross-check against the assembler in a test (the assembler is switched over by E05-02, not here).

## Tests

- [ ] `ipu-isa/test/test_slot_codec.py`: for every slot and instruction, encode with operand values at both ends of each domain, decode, and compare; `nop()` equals today's NOP encoding; unused opcodes raise `ReservedEncoding`.
- [ ] Cross-check (in `ipu-as-py/test/`): for every word of the E02-01 corpus, `WordCodec.unpack` then re-`pack` gives the same word, and each slot's `encode` equals what `Inst.encode` produced.
- [ ] E02 baselines unchanged (no production caller changes in this issue except tests).

## Acceptance Criteria

- [ ] Encoding and decoding of every slot is computed from the layout in `ipu_isa`, with no per-slot code.
- [ ] The interface covers what E05-02 lists; any difference is written into E05-02 before merge.
