"""Turn one parsed compound instruction into its word, through the ``ipu_isa`` codec (E05-02).

The assembler owns text: which slot each instruction takes, what each operand
token means, and where an error is. The bits are the codec's: each slot value
comes from ``SLOT_CODECS[slot].encode`` and the word from ``WORD_CODEC.pack``.

    words = [encode_word(compound) for compound in parsed]
    text = disassemble_word(word)
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import ipu_as.ipu_token as ipu_token
from ipu_as.errors import OperandError, SlotError
from ipu_as.inst import OPERAND_TYPE_MAP
from ipu_isa.codec.slot import OPCODE_SLOT, SLOT_CODECS, WORD_CODEC
from ipu_isa.codec.slot import EncodeError as CodecError
from ipu_isa.codec.word import WORD_LAYOUT, DecodedWord, FieldSpec
from ipu_isa.slots import LAYOUT_ORDER, SLOTS, Slot
from ipu_isa.view import isa

__all__ = [
    "LEGACY_SLOT_NAME",
    "SLOT_INSTANCES",
    "WILDCARD_OPCODES",
    "Placed",
    "assign_slots",
    "disassemble_word",
    "encode_word",
]

#: Mnemonics (lower case) that take the first free slot in layout order. Every
#: other instruction takes a free slot of its own kind. The editor takes both
#: rules from isa.json (``slotFill``).
WILDCARD_OPCODES = frozenset({"nop"})

#: Every slot instance of a word, most significant first: the order slots are
#: filled in, and the order of the disassembly text.
SLOT_INSTANCES: tuple[tuple[Slot, int], ...] = tuple(
    (slot, instance) for slot in LAYOUT_ORDER for instance in range(SLOTS[slot].count)
)

#: The name slot errors give each slot. It was the assembler's per-slot class
#: name (``LrInst``), and the editor matches on the message text, so it stays.
LEGACY_SLOT_NAME: Mapping[Slot, str] = MappingProxyType(
    {slot: SLOTS[slot].asm_opcode_class.removesuffix("Opcode") for slot in LAYOUT_ORDER}
)


@dataclass(frozen=True)
class Placed:
    """One instruction of a compound, in the slot instance it takes."""

    slot: Slot
    instance: int
    instruction: dict


@functools.cache
def _operand_classes(slot: Slot, mnemonic: str) -> tuple[type[ipu_token.IpuToken], ...]:
    """The token class of each operand of ``mnemonic`` in ``slot``, in assembly order."""
    instruction = SLOT_CODECS[slot].instruction(mnemonic)
    return tuple(OPERAND_TYPE_MAP[op.type.name] for op in instruction.operands)


def _slot_of(opcode: ipu_token.AnnotatedToken) -> Slot:
    slot = OPCODE_SLOT.get(opcode.token.value.upper())
    if slot is None:
        raise SlotError(
            f"Opcode '{opcode.token.value}' not found in any Inst subclass.",
            location=opcode.location(),
            token=opcode.token.value,
        )
    return slot


def _free_instance(taken: set[tuple[Slot, int]], slot: Slot | None) -> tuple[Slot, int] | None:
    """The first free instance of ``slot``, or of any slot when ``slot`` is None."""
    for key in SLOT_INSTANCES:
        if key not in taken and (slot is None or key[0] is slot):
            return key
    return None


def _place(instruction: dict, taken: set[tuple[Slot, int]]) -> tuple[Slot, int]:
    """The slot instance ``instruction`` takes, given the ones already ``taken``."""
    opcode = instruction["opcode"]
    if opcode.token.value.lower() in WILDCARD_OPCODES:
        key = _free_instance(taken, None)
        if key is None:
            raise SlotError(
                f"NOP: all slots are already filled\nAt: {opcode.get_location_string()}",
                location=opcode.location(),
                token=opcode.token.value,
            )
        return key
    slot = _slot_of(opcode)
    key = _free_instance(taken, slot)
    if key is None:
        name = LEGACY_SLOT_NAME[slot]
        raise SlotError(
            f"Too many instructions of type {name} (max {SLOTS[slot].count})\n"
            f"At: {opcode.get_location_string()}",
            location=opcode.location(),
            token=opcode.token.value,
        )
    return key


def _placements(compound: dict) -> Iterator[Placed]:
    """Each instruction of a parsed compound in the slot instance it takes, in order.

    Lazy, so the caller checks one instruction's operands before the next one
    is placed: errors come in source order.
    """
    taken: set[tuple[Slot, int]] = set()
    for instruction in compound["instructions"]:
        key = _place(instruction, taken)
        taken.add(key)
        yield Placed(*key, instruction)


def assign_slots(compound: dict) -> tuple[Placed, ...]:
    """Each instruction of a parsed compound, in the slot instance it takes.

    An instruction takes the first free instance of its slot; ``NOP`` takes the
    first free instance of any slot.

    Raises:
        SlotError: an unknown mnemonic, or more instructions than the slot has
            instances.
    """
    return tuple(_placements(compound))


def _operand_values(placed: Placed) -> tuple[int, ...]:
    """The encoded operands of one placed instruction, validated by their token classes."""
    opcode = placed.instruction["opcode"]
    operands = placed.instruction["operands"]
    classes = _operand_classes(placed.slot, opcode.token.value)
    if len(operands) != len(classes):
        raise OperandError(
            f"Instruction {opcode.token.value} expects {len(classes)} operands, "
            f"got {len(operands)}, in Line {opcode.token.line}, Column {opcode.token.column}.",
            location=opcode.location(),
            token=opcode.token.value,
        )
    tokens = [token_class(op) for token_class, op in zip(classes, operands, strict=True)]
    return tuple(token.encode() for token in tokens)


def encode_word(compound: dict) -> int:
    """The instruction word of one parsed compound (from ``ASTBuilder``).

    Raises:
        SlotError: see ``assign_slots``.
        OperandError: a wrong operand count or an invalid operand value.
        LabelError: an undefined label.

    Warns:
        UserWarning: an instruction of a simulation-only slot.
    """
    values = {}
    for placed in _placements(compound):
        operands = _operand_values(placed)
        opcode = placed.instruction["opcode"]
        try:
            values[(placed.slot, placed.instance)] = SLOT_CODECS[placed.slot].encode(
                opcode.token.value, operands
            )
        except CodecError as error:
            raise OperandError(
                str(error), location=opcode.location(), token=opcode.token.value
            ) from error
        if opcode.token.value.lower() not in WILDCARD_OPCODES and SLOTS[placed.slot].debug_only:
            warnings.warn(
                f"{opcode.token.value} uses the simulation-only {placed.slot.value} slot "
                "(not implemented in real IPU hardware)",
                stacklevel=2,
            )
    return WORD_CODEC.pack(values)


# -- disassembly text -------------------------------------------------------------


_Decoder = Callable[[int], str]

#: Each slot instance's fields, opcode first.
_INSTANCE_FIELDS: Mapping[tuple[Slot, int], tuple[FieldSpec, ...]] = MappingProxyType(
    {
        key: tuple(
            sorted((f for f in WORD_LAYOUT if (f.slot, f.instance) == key), key=lambda f: f.index)
        )
        for key in SLOT_INSTANCES
    }
)


@functools.cache
def _field_decoders(slot: Slot, opcode: int) -> tuple[_Decoder, ...]:
    """How to print each union field of ``slot`` when its opcode is ``opcode``.

    A field an operand of the instruction occupies prints as that operand's
    type; any other field prints as the field's canonical type.

    Raises:
        IndexError: ``opcode`` is no instruction of the slot.
    """
    view = isa()
    instruction = view.instructions(slot)[opcode]
    union = view.layout.unions[slot]
    types = [field.canonical_type for field in union.fields]
    by_name = {op.name: op.type.name for op in instruction.operands}
    for index, name in union.opcode_bindings[instruction.mnemonic]:
        types[index] = by_name[name]
    return tuple(OPERAND_TYPE_MAP[name].decode for name in types)


def _instance_text(fields: DecodedWord, key: tuple[Slot, int]) -> str:
    specs = _INSTANCE_FIELDS[key]
    opcode = fields[specs[0].name]
    mnemonic = isa().instructions(key[0])[opcode].mnemonic
    decoders = _field_decoders(key[0], opcode)
    operands = (decode(fields[spec.name]) for decode, spec in zip(decoders, specs[1:], strict=True))
    return " ".join((mnemonic, *operands))


def disassemble_word(word: int) -> str:
    """The assembly text of one word: every slot instance, NOP and unused fields included.

    Raises:
        IndexError: a slot holds an opcode no instruction uses.
    """
    fields = DecodedWord.from_word(word)
    return ";\n\t\t\t".join(_instance_text(fields, key) for key in SLOT_INSTANCES) + ";;"
