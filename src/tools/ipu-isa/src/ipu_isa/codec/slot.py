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
from ipu_isa.faults import FaultId, reserved_operand_codes
from ipu_isa.schema import SpecError
from ipu_isa.slots import Slot
from ipu_isa.view import Instruction, Isa, isa


class EncodeError(IpuError, ValueError):
    """An instruction cannot be encoded: unknown mnemonic, wrong operand count,
    an operand value outside its type's encodings or a reserved code, or a slot
    value that does not fit its slot. The message names the operand."""


class ReservedEncoding(IpuError, ValueError):
    """A slot value whose opcode no instruction uses: fault F-1 on hardware."""

    fault = FaultId.F1


@dataclass(frozen=True, slots=True)
class DecodedSlot:
    """One slot instance, decoded.

    Attributes:
        slot: The slot it was decoded from.
        instruction: The instruction its opcode selects.
        operands: Encoded operand values, in assembly operand order.
        unused_bits: Mask of the non-zero bits that neither the opcode nor any
            operand of ``instruction`` occupies.
        value: The slot's bits as decoded.
    """

    slot: Slot
    instruction: Instruction
    operands: tuple[int, ...]
    unused_bits: int
    value: int


@dataclass(frozen=True, slots=True)
class _Field:
    shift: int  # LSB position within the slot value
    bits: int  # column width


@dataclass(frozen=True, slots=True)
class _Operand:
    name: str
    field: _Field
    bits: int  # the operand's own width, LSB-aligned in its column
    reserved: frozenset[int]  # codes with no meaning (F-1), from ipu_isa.faults

    @property
    def mask(self) -> int:
        return ((1 << self.bits) - 1) << self.field.shift


class SlotCodec:
    """Encoder and decoder of one slot; one instance serves every copy of it.

    Attributes:
        slot: The slot this codec encodes.
        instances: How many copies of the slot one word holds.
        bits: The slot's width.
    """

    __slots__ = (
        "_slot",
        "_instances",
        "_bits",
        "_fields",
        "_opcode_shift",
        "_instructions",
        "_by_mnemonic",
        "_operands",
        "_used",
        "_nop",
    )

    def __init__(self, view: Isa, slot: Slot) -> None:
        union = view.layout.unions[slot]
        self._slot = slot
        self._instances = view.slots[slot].count
        self._bits = union.bits
        fields: list[_Field] = []
        shift = self.bits - union.opcode_bits
        for field in union.fields:
            shift -= field.bits
            fields.append(_Field(shift, field.bits))
        self._fields = tuple(fields)
        self._opcode_shift = self.bits - union.opcode_bits
        self._instructions = view.instructions(slot)
        self._by_mnemonic = {i.mnemonic.upper(): i for i in self._instructions}
        self._operands: dict[str, tuple[_Operand, ...]] = {}
        self._used: dict[str, int] = {}  # mnemonic -> mask of its opcode and operand bits
        opcode_mask = ((1 << union.opcode_bits) - 1) << self._opcode_shift
        for instruction in self._instructions:
            bindings = dict(
                (name, index) for index, name in union.opcode_bindings[instruction.mnemonic]
            )
            operands = []
            for op in instruction.operands:
                bits = view.layout.operand_bits[op.type.name]
                reserved = frozenset(reserved_operand_codes(op.type, bits))
                operands.append(_Operand(op.name, fields[bindings[op.name]], bits, reserved))
            self._operands[instruction.mnemonic] = tuple(operands)
            used = opcode_mask
            for operand in operands:
                used |= operand.mask
            self._used[instruction.mnemonic] = used
        self._nop = self.encode("NOP", ())

    @property
    def slot(self) -> Slot:
        return self._slot

    @property
    def instances(self) -> int:
        return self._instances

    @property
    def bits(self) -> int:
        return self._bits

    def instruction(self, mnemonic: str) -> Instruction:
        """The slot's instruction ``mnemonic`` (case-insensitive).

        Raises:
            EncodeError: The slot has no such instruction.
        """
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
                outside its operand's field or one of its reserved codes.
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
            if not 0 <= encoded < 1 << operand.bits:
                raise EncodeError(
                    f"{self.slot}.{instruction.mnemonic}: operand {operand.name} = {encoded} "
                    f"is outside 0..{(1 << operand.bits) - 1}"
                )
            if encoded in operand.reserved:
                raise EncodeError(
                    f"{self.slot}.{instruction.mnemonic}: operand {operand.name} = {encoded} "
                    "is a reserved code"
                )
            value |= encoded << operand.field.shift
        return value

    def nop(self) -> int:
        """The slot value of ``NOP``."""
        return self._nop

    def field_values(self, value: int) -> tuple[int, ...]:
        """The raw value of every union field of a slot value.

        Args:
            value: The slot's bits.

        Returns:
            Each field's bits, field 0 first, bound to an operand or not.
        """
        return tuple((value >> field.shift) & ((1 << field.bits) - 1) for field in self._fields)

    def decode(self, value: int) -> DecodedSlot:
        """Decode one slot value.

        Operand codes are returned as stored; a reserved operand code is not
        rejected here.

        Args:
            value: The slot's bits, ``0 <= value < 2**bits``.

        Returns:
            The instruction, its encoded operands and any stray bits.

        Raises:
            EncodeError: ``value`` does not fit the slot.
            ReservedEncoding: The opcode belongs to no instruction.
        """
        if not 0 <= value < 1 << self._bits:
            raise EncodeError(f"{self._slot}: value {value} does not fit {self._bits} bits")
        opcode = value >> self._opcode_shift
        if opcode >= len(self._instructions):
            raise ReservedEncoding(f"{self._slot}: opcode {opcode} is reserved")
        instruction = self._instructions[opcode]
        operands = tuple(
            (value & operand.mask) >> operand.field.shift
            for operand in self._operands[instruction.mnemonic]
        )
        return DecodedSlot(
            self._slot, instruction, operands, value & ~self._used[instruction.mnemonic], value
        )


class WordCodec:
    """Packs slot values into an instruction word and unpacks them."""

    def __init__(self, view: Isa, slot_codecs: Mapping[Slot, SlotCodec]) -> None:
        self._codecs = slot_codecs
        lowest: dict[tuple[Slot, int], int] = {}
        for field in word_layout(view):
            lowest.setdefault((field.slot, field.instance), field.shift)
        # (slot, instance) -> (LSB of that instance, slot width, NOP value)
        self._instances = MappingProxyType(
            {
                key: (shift, slot_codecs[key[0]].bits, slot_codecs[key[0]].nop())
                for key, shift in lowest.items()
            }
        )

    def pack(self, slot_values: Mapping[tuple[Slot, int], int]) -> int:
        """The word holding ``slot_values``.

        Args:
            slot_values: (slot, instance) -> slot value. A missing instance is NOP.

        Returns:
            The instruction word.

        Raises:
            EncodeError: An unknown (slot, instance), or a value that does not
                fit its slot.
        """
        unknown = set(slot_values) - set(self._instances)
        if unknown:
            raise EncodeError(f"no such slot instance(s): {sorted(map(str, unknown))}")
        word = 0
        for key, (shift, bits, nop) in self._instances.items():
            value = slot_values.get(key, nop)
            if not 0 <= value < 1 << bits:
                raise EncodeError(f"{key[0]}[{key[1]}]: value {value} does not fit {bits} bits")
            word |= value << shift
        return word

    def unpack(self, word: int) -> dict[tuple[Slot, int], DecodedSlot]:
        """Every slot instance of ``word``, decoded.

        Raises:
            ReservedEncoding: A slot holds an opcode no instruction uses.
        """
        return {
            key: self._codecs[key[0]].decode((word >> shift) & ((1 << bits) - 1))
            for key, (shift, bits, _) in self._instances.items()
        }


@dataclass(frozen=True, slots=True)
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
            try:
                opcode_slot[mnemonic] = view.find_slot(mnemonic)
            except ValueError as error:
                raise SpecError(str(error)) from None
    return Codecs(MappingProxyType(slots), WordCodec(view, slots), MappingProxyType(opcode_slot))


def codecs(view: Isa | None = None) -> Codecs:
    """The codecs of an ISA, built once per ISA.

    Args:
        view: The ISA; default ``isa()``.

    Returns:
        Its slot codecs, word codec and mnemonic -> slot map.
    """
    return _codecs(isa() if view is None else view)


#: The repository ISA's slot codecs.
SLOT_CODECS: Mapping[Slot, SlotCodec] = codecs().slots
#: The repository ISA's word codec.
WORD_CODEC: WordCodec = codecs().word
#: Upper-case mnemonic -> slot (NOP excluded: it is in every slot).
OPCODE_SLOT: Mapping[str, Slot] = codecs().opcode_slot
