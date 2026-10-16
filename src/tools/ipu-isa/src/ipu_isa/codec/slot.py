"""Encode and decode one instruction in its slot, from the union layout (E03-17).

A slot instance is the opcode in its top bits and the union fields below it,
field 0 just under the opcode and the last field in the lowest bits. An
instruction places each operand in the field its ``opcode_bindings`` names,
LSB-aligned; every bit it does not use is 0.

Operand values here are already integers, as encoded: an LR is its index, an
``LrModPow2KImmediate`` is ``k - 1``. Turning text into those integers is the
assembler's job (E05-02).
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from ipu_isa.codec.word import word_layout
from ipu_isa.errors import IpuError
from ipu_isa.slots import Slot
from ipu_isa.view import Instruction, Isa, isa


class EncodeError(IpuError, ValueError):
    """An instruction cannot be encoded: unknown mnemonic, wrong operand count,
    or an operand value outside its type's encodings. The message names the
    operand."""


class ReservedEncoding(IpuError, ValueError):
    """A slot value whose opcode no instruction uses (fault F-1 on hardware)."""


@dataclass(frozen=True, slots=True)
class DecodedSlot:
    """One slot instance, decoded."""

    slot: Slot
    instruction: Instruction
    operands: tuple[int, ...]  # encoded values, in assembly operand order
    unused_bits: int  # mask of the non-zero bits no operand of this instruction uses


@dataclass(frozen=True, slots=True)
class _Field:
    shift: int  # LSB position within the slot value
    bits: int  # column width


@dataclass(frozen=True, slots=True)
class _Operand:
    name: str
    field: _Field
    bits: int  # the operand's own width, LSB-aligned in its column
    encodings: int  # how many values it may take


class SlotCodec:
    """Encoder and decoder of one slot; one instance serves every copy of it."""

    def __init__(self, view: Isa, slot: Slot) -> None:
        union = view.layout.unions[slot]
        self.slot = slot
        self.instances = view.slots[slot].count
        self.bits = union.bits
        fields: list[_Field] = []
        shift = self.bits - union.opcode_bits
        for field in union.fields:
            shift -= field.bits
            fields.append(_Field(shift, field.bits))
        self._opcode_shift = self.bits - union.opcode_bits
        self._instructions = view.instructions(slot)
        self._by_mnemonic = {i.mnemonic.upper(): i for i in self._instructions}
        self._operands: dict[str, tuple[_Operand, ...]] = {}
        for instruction in self._instructions:
            bindings = dict(
                (name, index) for index, name in union.opcode_bindings[instruction.mnemonic]
            )
            self._operands[instruction.mnemonic] = tuple(
                _Operand(
                    op.name,
                    fields[bindings[op.name]],
                    view.layout.operand_bits[op.type.name],
                    len(op.type.domain.members) or 1 << view.layout.operand_bits[op.type.name],
                )
                for op in instruction.operands
            )

    def instruction(self, mnemonic: str) -> Instruction:
        """The slot's instruction ``mnemonic`` (case-insensitive)."""
        try:
            return self._by_mnemonic[mnemonic.upper()]
        except KeyError:
            raise EncodeError(f"{self.slot}: no instruction {mnemonic!r}") from None

    def encode(self, mnemonic: str, operands: Sequence[int]) -> int:
        """The slot value of ``mnemonic`` with encoded ``operands``.

        Args:
            mnemonic: An instruction of this slot (case-insensitive).
            operands: Encoded operand values, in assembly order.

        Returns:
            The slot's bits.

        Raises:
            EncodeError: Unknown mnemonic, wrong operand count, or a value
                outside its operand's encodings.
        """
        instruction = self.instruction(mnemonic)
        layout = self._operands[instruction.mnemonic]
        if len(operands) != len(layout):
            raise EncodeError(
                f"{self.slot}.{instruction.mnemonic}: takes {len(layout)} operand(s), "
                f"got {len(operands)}"
            )
        value = instruction.opcode << self._opcode_shift
        for operand, encoded in zip(layout, operands, strict=True):
            if not 0 <= encoded < operand.encodings:
                raise EncodeError(
                    f"{self.slot}.{instruction.mnemonic}: operand {operand.name} = {encoded} "
                    f"is outside 0..{operand.encodings - 1}"
                )
            value |= encoded << operand.field.shift
        return value

    def nop(self) -> int:
        """The slot value of ``NOP``."""
        return self.encode("NOP", ())

    def decode(self, value: int) -> DecodedSlot:
        """Decode one slot value.

        Raises:
            ReservedEncoding: The opcode belongs to no instruction.
        """
        opcode = value >> self._opcode_shift
        if opcode >= len(self._instructions):
            raise ReservedEncoding(f"{self.slot}: opcode {opcode} is reserved")
        instruction = self._instructions[opcode]
        used = ((1 << (self.bits - self._opcode_shift)) - 1) << self._opcode_shift
        operands = []
        for operand in self._operands[instruction.mnemonic]:
            mask = (1 << operand.bits) - 1
            operands.append((value >> operand.field.shift) & mask)
            used |= mask << operand.field.shift
        return DecodedSlot(self.slot, instruction, tuple(operands), value & ~used)


class WordCodec:
    """Packs slot values into an instruction word and unpacks them."""

    def __init__(self, view: Isa, slot_codecs: Mapping[Slot, SlotCodec]) -> None:
        self._codecs = slot_codecs
        lowest: dict[tuple[Slot, int], int] = {}
        for field in word_layout(view):
            lowest.setdefault((field.slot, field.instance), field.shift)
        self._shifts = MappingProxyType(lowest)  # (slot, instance) -> LSB of that instance

    def pack(self, slot_values: Mapping[tuple[Slot, int], int]) -> int:
        """The word holding ``slot_values``; a missing (slot, instance) is NOP."""
        unknown = set(slot_values) - set(self._shifts)
        if unknown:
            raise EncodeError(f"no such slot instance(s): {sorted(map(str, unknown))}")
        word = 0
        for key, shift in self._shifts.items():
            codec = self._codecs[key[0]]
            value = slot_values.get(key, codec.nop())
            if not 0 <= value < 1 << codec.bits:
                raise EncodeError(f"{key[0]}[{key[1]}]: value is wider than {codec.bits} bits")
            word |= value << shift
        return word

    def unpack(self, word: int) -> dict[tuple[Slot, int], DecodedSlot]:
        """Every slot instance of ``word``, decoded."""
        return {
            key: self._codecs[key[0]].decode(
                (word >> shift) & ((1 << self._codecs[key[0]].bits) - 1)
            )
            for key, shift in self._shifts.items()
        }


@dataclass(frozen=True)
class Codecs:
    """Every codec of one ISA."""

    slots: Mapping[Slot, SlotCodec]
    word: WordCodec
    #: Upper-case mnemonic -> its slot. NOP is in every slot, so it is absent.
    opcode_slot: Mapping[str, Slot]


@functools.cache
def _codecs(view: Isa) -> Codecs:
    slots = {slot: SlotCodec(view, slot) for slot in view.layout.layout_order}
    opcode_slot: dict[str, Slot] = {}
    for slot in view.layout.layout_order:
        for instruction in view.instructions(slot):
            mnemonic = instruction.mnemonic.upper()
            if mnemonic == "NOP":
                continue
            if mnemonic in opcode_slot:
                raise EncodeError(f"{mnemonic} is in slots {opcode_slot[mnemonic]} and {slot}")
            opcode_slot[mnemonic] = slot
    return Codecs(MappingProxyType(slots), WordCodec(view, slots), MappingProxyType(opcode_slot))


def codecs(view: Isa | None = None) -> Codecs:
    """The codecs of ``view`` (default: ``isa()``), built once per ISA."""
    return _codecs(isa() if view is None else view)


#: The repository ISA's slot codecs.
SLOT_CODECS: Mapping[Slot, SlotCodec] = codecs().slots
#: The repository ISA's word codec.
WORD_CODEC: WordCodec = codecs().word
#: Upper-case mnemonic -> slot (NOP excluded: it is in every slot).
OPCODE_SLOT: Mapping[str, Slot] = codecs().opcode_slot
