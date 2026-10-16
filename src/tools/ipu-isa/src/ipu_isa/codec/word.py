"""The instruction word: field layout, decode, encode and program bytes (E03-16).

``WORD_LAYOUT`` lists every field of the emulator word, LSB first, with its
wire name, slot, instance, position and width. It is built from the slot and
operand-type registries and the union layout, never from Python class names:

* a slot instance is ``SlotInfo.field_prefix`` (``lr_inst_0`` when the slot
  appears more than once);
* field 0 of an instance is its opcode, named after ``SlotInfo.opcode_enum``;
  field ``k`` is union field ``k - 1``, named after its canonical operand
  type's ``OperandType.field_suffix``;
* slots are in ``LAYOUT_ORDER`` (most significant first), instances in order,
  and within an instance the opcode is the most significant field.

A program file holds one word per instruction, each little-endian and padded
to a multiple of 32 bits (``word_bytes``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ipu_isa.schema import SpecError
from ipu_isa.slots import Slot
from ipu_isa.tags import Tag
from ipu_isa.view import Isa, isa

#: Program files pad each word to a multiple of this many bits.
_ALIGN_BITS = 32


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """One field of the instruction word."""

    name: str  # "lr_inst_0_token_0_lr_inst_opcode": wire format
    slot: Slot
    instance: int  # 0 .. count - 1
    index: int  # 0 = opcode, k = union field k - 1
    bits: int
    shift: int  # LSB position in the emulator word


def _instance_prefix(view: Isa, slot: Slot, instance: int) -> str:
    info = view.slots[slot]
    return info.field_prefix if info.count == 1 else f"{info.field_prefix}_{instance}"


def _field_name(view: Isa, slot: Slot, instance: int, index: int) -> str:
    info = view.slots[slot]
    if index == 0:
        suffix = info.opcode_enum
    else:
        canonical = view.layout.unions[slot].fields[index - 1].canonical_type
        suffix = view.operand_types[canonical].field_suffix
    return f"{_instance_prefix(view, slot, instance)}_token_{index}_{suffix}"


def word_layout(view: Isa | None = None) -> tuple[FieldSpec, ...]:
    """Every field of ``view``'s word (default: ``isa()``), LSB first."""
    view = isa() if view is None else view
    fields: list[FieldSpec] = []
    shift = 0
    for slot in reversed(view.layout.layout_order):
        union = view.layout.unions[slot]
        widths = [union.opcode_bits, *(field.bits for field in union.fields)]
        for instance in reversed(range(view.slots[slot].count)):
            for index in reversed(range(len(widths))):
                name = _field_name(view, slot, instance, index)
                fields.append(FieldSpec(name, slot, instance, index, widths[index], shift))
                shift += widths[index]
    if shift != view.layout.word_bits():
        raise SpecError(f"word layout is {shift} bits, the ISA says {view.layout.word_bits()}")
    return tuple(fields)


#: Every field of the repository ISA's word, LSB first.
WORD_LAYOUT: tuple[FieldSpec, ...] = word_layout()

_BY_POSITION = {(f.slot, f.instance, f.index): f for f in WORD_LAYOUT}
_BY_INSTANCE: dict[tuple[Slot, int], tuple[FieldSpec, ...]] = {
    key: tuple(f for f in WORD_LAYOUT if (f.slot, f.instance) == key)
    for key in dict.fromkeys((f.slot, f.instance) for f in WORD_LAYOUT)
}

# (name, shift, mask) of every field, for the decode and encode loops.
_FIELDS = tuple((f.name, f.shift, (1 << f.bits) - 1) for f in WORD_LAYOUT)


def field_name(slot: Slot, instance: int, index: int) -> str:
    """The wire name of field ``index`` of ``slot`` instance ``instance``."""
    return _BY_POSITION[(slot, instance, index)].name


def opcode_field(slot: Slot, instance: int = 0) -> str:
    """The wire name of the opcode field of ``slot`` instance ``instance``."""
    return field_name(slot, instance, 0)


class DecodedWord(dict[str, int]):
    """A decoded instruction word: wire field name -> value.

    A ``dict`` so emulator dispatch (``inst[key]``) and existing callers keep
    working; a field missing from it encodes as 0.
    """

    @classmethod
    def from_word(cls, word: int) -> DecodedWord:
        """Split ``word`` into its fields.

        Args:
            word: An emulator word.

        Returns:
            Every field of ``WORD_LAYOUT``, by wire name.
        """
        return cls((name, (word >> shift) & mask) for name, shift, mask in _FIELDS)

    def to_word(self) -> int:
        """Pack the fields back into a word; a field this dict lacks is 0.

        Returns:
            The emulator word.
        """
        word = 0
        for name, shift, mask in _FIELDS:
            word |= (self.get(name, 0) & mask) << shift
        return word

    def slot_value(self, slot: Slot, instance: int = 0) -> int:
        """The bits of one slot instance, as that slot encodes on its own."""
        fields = _BY_INSTANCE[(slot, instance)]
        lowest = fields[0].shift  # LSB first
        value = 0
        for f in fields:
            value |= (self.get(f.name, 0) & ((1 << f.bits) - 1)) << (f.shift - lowest)
        return value


def word_bytes(tag: Tag | None = None) -> int:
    """Bytes per word in a program file, padded to 32 bits.

    Args:
        tag: ``None`` for the whole emulator word (24 bytes), ``Tag.HARDWARE``
            for the hardware word (20 bytes); ``Tag.EMULATOR`` gives the width
            of the EMULATOR slots alone.

    Returns:
        The byte count.
    """
    bits = isa().layout.word_bits(tag)
    return -(-bits // _ALIGN_BITS) * (_ALIGN_BITS // 8)


def hardware_word(word: int) -> int:
    """The hardware part of an emulator word: its top ``word_bits(HARDWARE)``
    bits. The slot registry rejects a HARDWARE slot after an EMULATOR one
    (E03-07, Q41), so the EMULATOR slots are the least significant bits."""
    return word >> isa().layout.word_bits(Tag.EMULATOR)


def unpack_words(data: bytes) -> list[int]:
    """The words of a program file; a trailing partial word is ignored.

    Args:
        data: Program file bytes (``pack_program``'s layout).

    Returns:
        One emulator word per instruction.
    """
    width = word_bytes()
    return [
        int.from_bytes(data[offset : offset + width], "little")
        for offset in range(0, len(data) - width + 1, width)
    ]


def unpack_program(data: bytes) -> list[DecodedWord]:
    """Decode a program file; a trailing partial word is ignored."""
    return [DecodedWord.from_word(word) for word in unpack_words(data)]


def pack_program(words: Sequence[int]) -> bytes:
    """The program file bytes of ``words``."""
    width = word_bytes()
    return b"".join(word.to_bytes(width, byteorder="little") for word in words)
