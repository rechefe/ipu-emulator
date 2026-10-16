"""Instruction words to assembly text, from the slot codec (E03-18).

The one place disassembly text is made. ``disassemble`` writes every field of
every slot, bound to an operand or not, in the format the assembler's
``CompoundInst.decode`` always had; ``slot_operations`` writes each slot's
mnemonic and its operands only, in assembly operand order, for the debugger's
pipeline view.
"""

from __future__ import annotations

from ipu_isa.codec.slot import DecodedSlot, codecs
from ipu_isa.operand_types import OperandKind, OperandType
from ipu_isa.reshape_mask import RESHAPE_MASK_LR_OFFSET
from ipu_isa.view import Isa, Operand, isa

#: Between two slots of one word, and after the last.
SLOT_SEPARATOR = ";\n\t\t\t"
WORD_END = ";;"


def type_text(operand_type: OperandType, value: int) -> str:
    """The assembly text of a field value of ``operand_type``.

    * REGISTER and ENUM: the member name (``lr3``, ``on_inv``); a value past
      the last member as ``<illegal_<type>_<value>>``.
    * ``LrOrReshapeMaskImmediate``: ``lr<n>`` from ``RESHAPE_MASK_LR_OFFSET``
      up, the number below it.
    * An immediate or label whose value range fits its field is stored from
      code 0 upward (``k - 1`` for ``INCR_MOD_POW2``), so the text adds the
      range's low end back. Any other value is written as stored.
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
    """The assembly text of operand ``op`` holding the encoded ``value``."""
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
        The slot's text, unused fields included, as ``CompoundInst.decode``
        always wrote it.
    """
    view = _view(isa)
    return " ".join([decoded.instruction.mnemonic, *_field_texts(decoded, view)])


def disassemble(word: int, *, isa: Isa | None = None) -> str:
    """The assembly text of one instruction word.

    Args:
        word: The emulator instruction word.
        isa: The ISA; default ``ipu_isa.isa()``.

    Returns:
        Every slot instance's ``slot_text`` in layout order (MSB first),
        joined by ``SLOT_SEPARATOR`` and ended by ``WORD_END``.

    Raises:
        ReservedEncoding: A slot holds an opcode no instruction uses.
    """
    view = _view(isa)
    return (
        SLOT_SEPARATOR.join(slot_text(decoded, isa=view) for _, decoded in _slots(word, view))
        + WORD_END
    )


def slot_operations(word: int, *, isa: Isa | None = None) -> list[tuple[str, str]]:
    """``(instance label, "MNEMONIC op op ...")`` for every slot instance.

    Args:
        word: The emulator instruction word.
        isa: The ISA; default ``ipu_isa.isa()``.

    Returns:
        One pair per slot instance in layout order (MSB first). The label is
        ``SlotInfo.instance_labels`` (``LR0``, ``MULT``); the text holds the
        operands only, in assembly operand order.

    Raises:
        ReservedEncoding: A slot holds an opcode no instruction uses.
    """
    view = _view(isa)
    result = []
    for label, decoded in _slots(word, view):
        bindings = view.layout.unions[decoded.slot].opcode_bindings[decoded.instruction.mnemonic]
        fields = _field_texts(decoded, view)
        operands = [fields[index] for index, _ in bindings]
        result.append((label, " ".join([decoded.instruction.mnemonic, *operands])))
    return result


def _view(view: Isa | None) -> Isa:
    return isa() if view is None else view


def _slots(word: int, view: Isa) -> list[tuple[str, DecodedSlot]]:
    unpacked = codecs(view).word.unpack(word)
    return [
        (label, unpacked[(slot, instance)])
        for slot in view.layout.layout_order
        for instance, label in enumerate(view.slots[slot].instance_labels())
    ]
