"""Instruction words to assembly text, from the slot codec (E03-18).

The one place disassembly text is made.

* ``disassemble``: every slot instance in layout order (MSB first), each as
  ``MNEMONIC f0 f1 ...`` with every union field, bound to an operand or not,
  joined by ``SLOT_SEPARATOR`` and ended by ``WORD_END``. For the hardware
  target (E05-15) the word is the hardware word and only hardware slots print.
  ``disassemble_fields`` does the same for a word already split into fields.
* ``operation_text``: one slot's mnemonic and its operands only, in assembly
  operand order; ``slot_operations`` labels it per slot instance, for the
  debugger's pipeline view.
"""

from __future__ import annotations

from collections.abc import Mapping

from ipu_isa.codec.slot import DecodedSlot, codecs
from ipu_isa.codec.word import DecodedWord
from ipu_isa.errors import IsaValueError
from ipu_isa.operand_types import OperandKind, OperandType
from ipu_isa.reshape_mask import RESHAPE_MASK_LR_OFFSET
from ipu_isa.tags import Tag
from ipu_isa.view import Isa, Operand
from ipu_isa.view import isa as default_isa

#: Between two slots of one word, and after the last.
SLOT_SEPARATOR = ";\n\t\t\t"
WORD_END = ";;"

#: The words a program can be disassembled from.
TARGETS = (Tag.EMULATOR, Tag.HARDWARE)


def target_tag(target: Tag | str) -> Tag:
    """The word a target names.

    Args:
        target: ``"emulator"`` or ``"hardware"``, or the ``Tag`` itself.

    Returns:
        ``Tag.EMULATOR`` or ``Tag.HARDWARE``.

    Raises:
        ValueError: Any other target.
    """
    if target not in TARGETS:
        raise IsaValueError(f"target must be emulator or hardware, not {target!r}")
    return Tag(target)


def type_text(operand_type: OperandType, value: int) -> str:
    """The assembly text of a field value of ``operand_type``.

    Args:
        operand_type: How to read the value.
        value: The field's bits.

    Returns:
        The text:

    * REGISTER and ENUM: the member name (``lr3``, ``on_inv``); a value past
      the last member as ``<illegal_<type>_<value>>``.
    * ``LrOrReshapeMaskImmediate``: ``lr<n>`` from ``RESHAPE_MASK_LR_OFFSET``
      up, the number below it.
    * An immediate or label whose value range fits its field is stored from
      code 0 upward (``k - 1`` for ``INC.MODP2``), so the text adds the
      range's low end back. Any other value is written as stored. The width
      is the type's declared ``bits``; every ranged type declares one, so it
      equals the layout width ``ipu_isa.faults`` uses for the same rule.
    """
    domain = operand_type.domain
    if operand_type.kind in (OperandKind.REGISTER, OperandKind.ENUM):
        if 0 <= value < len(domain.members):
            return domain.members[value]
        return f"<illegal_{operand_type.name}_{value}>"
    if operand_type.name == "LrOrReshapeMaskImmediate":
        if value >= RESHAPE_MASK_LR_OFFSET:
            return f"lr{value - RESHAPE_MASK_LR_OFFSET}"
        return str(value)
    if domain.value_range is not None:
        low, high = domain.value_range
        bits = operand_type.bits
        if bits is not None and high - low + 1 <= 1 << bits:
            return str(value + low)
    return str(value)


def operand_text(op: Operand, value: int) -> str:
    """The assembly text of an operand.

    Args:
        op: The operand.
        value: Its encoded value.

    Returns:
        ``type_text`` of the operand's type.
    """
    return type_text(op.type, value)


def _field_texts(decoded: DecodedSlot, view: Isa) -> list[str]:
    """Every union field of ``decoded`` as text, field 0 first: a bound field in
    its operand's type, an unbound one in the field's canonical type."""
    union = view.layout.unions[decoded.slot]
    values = codecs(view).slots[decoded.slot].field_values(decoded.value)
    types = [view.operand_types[field.canonical_type] for field in union.fields]
    for index, operand in zip(
        (index for index, _ in union.opcode_bindings[decoded.instruction.mnemonic]),
        decoded.instruction.operands,
        strict=True,
    ):
        types[index] = operand.type
    return [type_text(t, v) for t, v in zip(types, values, strict=True)]


def slot_text(decoded: DecodedSlot, *, isa: Isa | None = None) -> str:
    """``MNEMONIC f0 f1 ...``: the mnemonic, then every field of the slot.

    Args:
        decoded: One decoded slot instance.
        isa: The ISA it was decoded with; default ``ipu_isa.isa()``.

    Returns:
        The slot's text, unused fields included.
    """
    view = _view(isa)
    return " ".join([decoded.instruction.mnemonic, *_field_texts(decoded, view)])


def disassemble(word: int, *, isa: Isa | None = None, target: Tag | str = Tag.EMULATOR) -> str:
    """The assembly text of one instruction word.

    Args:
        word: The emulator word, or for ``Tag.HARDWARE`` the hardware word
            (its top bits, ``codec.word.hardware_word``).
        isa: The ISA; default ``ipu_isa.isa()``.
        target: Which word ``word`` is; the hardware text has the hardware
            slots only.

    Returns:
        Each printed slot instance's ``slot_text`` in layout order (MSB
        first), joined by ``SLOT_SEPARATOR`` and ended by ``WORD_END``.

    Raises:
        ValueError: An unknown target.
        ReservedEncoding: A slot holds an opcode no instruction uses.
    """
    view = _view(isa)
    tag = target_tag(target)
    if tag is Tag.HARDWARE:
        word <<= view.layout.word_bits(Tag.EMULATOR)  # the emulator slots are the low bits
    slots = _slots(word, view, tag)
    return SLOT_SEPARATOR.join(slot_text(decoded, isa=view) for _, decoded in slots) + WORD_END


def disassemble_fields(
    fields: Mapping[str, int], *, isa: Isa | None = None, target: Tag | str = Tag.EMULATOR
) -> str:
    """``disassemble`` of a word already split into its fields (``DecodedWord``).

    Args:
        fields: Wire field name -> value; a missing field is 0, and a value
            wider than its field is cut to the field, as packing would.
        isa: The ISA; default ``ipu_isa.isa()``.
        target: Which slots to print: every slot, or the hardware slots.

    Returns:
        As ``disassemble`` of the emulator word.

    Raises:
        ValueError: An unknown target.
        ReservedEncoding: A slot holds an opcode no instruction uses.
    """
    view = _view(isa)
    tag = target_tag(target)
    slots = _slots(DecodedWord(fields).to_word(), view, tag)
    return SLOT_SEPARATOR.join(slot_text(decoded, isa=view) for _, decoded in slots) + WORD_END


def operation_text(decoded: DecodedSlot, *, isa: Isa | None = None) -> str:
    """One slot's instruction as written in assembly: the mnemonic, then its operands.

    Unlike ``slot_text``, operands are in assembly order and fields the
    instruction does not use are left out. Each operand prints its whole
    field, as ``slot_text`` does, so an illegal value shows as illegal.

    Args:
        decoded: A slot value decoded by its ``SlotCodec``.
        isa: The ISA it was decoded with; default ``ipu_isa.isa()``.

    Returns:
        ``"ADD lr0 lr1 lr2"``; a bare ``"NOP"``.
    """
    view = _view(isa)
    bindings = view.layout.unions[decoded.slot].opcode_bindings[decoded.instruction.mnemonic]
    fields = _field_texts(decoded, view)
    return " ".join((decoded.instruction.mnemonic, *(fields[index] for index, _ in bindings)))


def slot_operations(word: int, *, isa: Isa | None = None) -> list[tuple[str, str]]:
    """``(instance label, operation_text)`` for every slot instance.

    Args:
        word: The emulator instruction word.
        isa: The ISA; default ``ipu_isa.isa()``.

    Returns:
        One pair per slot instance in layout order (MSB first). The label is
        ``SlotInfo.instance_labels`` (``LR0``, ``MULT``).

    Raises:
        ReservedEncoding: A slot holds an opcode no instruction uses.
    """
    view = _view(isa)
    return [
        (label, operation_text(decoded, isa=view)) for label, decoded in _slots(word, view, None)
    ]


def _view(view: Isa | None) -> Isa:
    return default_isa() if view is None else view


def _slots(word: int, view: Isa, tag: Tag | None) -> list[tuple[str, DecodedSlot]]:
    """Each slot instance of ``word`` (of the ``tag`` slots, or all), in layout order."""
    unpacked = codecs(view).word.unpack(word)
    return [
        (label, unpacked[(slot, instance)])
        for slot in view.layout.layout_order
        if tag in (None, Tag.EMULATOR) or view.slots[slot].tag is tag
        for instance, label in enumerate(view.slots[slot].instance_labels())
    ]
