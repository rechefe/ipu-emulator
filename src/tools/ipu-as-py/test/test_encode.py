"""The assembler encodes through the ipu_isa codec (E05-02).

Every instruction of every slot round-trips: assembled alone, its slot decodes
(``SLOT_CODECS``) to the same instruction, and the operands printed back
assemble to the same word.
"""

from __future__ import annotations

import warnings

import lark
import pytest

import ipu_as.ipu_token as ipu_token
from ipu_as import encode
from ipu_as.errors import OperandError, SlotError
from ipu_as.inst import OPERAND_TYPE_MAP
from ipu_as.session import AssemblySession
from ipu_isa.codec.disasm import type_text
from ipu_isa.codec.slot import WORD_CODEC
from ipu_isa.codec.word import DecodedWord
from ipu_isa.slots import LAYOUT_ORDER, Slot
from ipu_isa.view import isa


def _word(source: str) -> int:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (word,) = AssemblySession(source).assemble().words
    return word


def _text(operand_type: str, value: int) -> str:
    text = type_text(isa().operand_types[operand_type], value)
    # A label decodes to its address; at address 0 that is the offset +N.
    return f"+{text}" if operand_type == "Label" else text


def _instructions():
    for slot in LAYOUT_ORDER:
        for instruction in isa().instructions(slot):
            if instruction.mnemonic != "NOP":
                yield pytest.param(slot, instruction, id=f"{slot.value}.{instruction.mnemonic}")


def _distinct_operand(token_class, position: int) -> str:
    """A valid operand text that is not the type's zero, varied by ``position``."""
    if issubclass(token_class, ipu_token.LabelToken):
        return f"+{position + 1}"
    if issubclass(token_class, ipu_token.EnumToken):
        members = [m for m in token_class.enum_array() if not m.lower().startswith("reserved")]
        return members[(position + 1) % len(members)]
    low, high = token_class.value_range()
    return str(min(high, max(low, position + 1)))


@pytest.mark.parametrize(("slot", "instruction"), _instructions())
def test_every_instruction_round_trips(slot, instruction):
    classes = [OPERAND_TYPE_MAP[op.type.name] for op in instruction.operands]
    operands = [_distinct_operand(cls, i) for i, cls in enumerate(classes)]
    word = _word(" ".join([instruction.mnemonic, *operands]) + ";;")
    decoded = WORD_CODEC.unpack(word)[(slot, 0)]
    assert decoded.instruction.mnemonic == instruction.mnemonic
    # Distinct values come back in assembly order: each lands in its own field.
    expected = [
        cls(ipu_token.AnnotatedToken(lark.Token("OPERAND", text), 0)).encode()
        for cls, text in zip(classes, operands, strict=True)
    ]
    assert list(decoded.operands) == expected
    assert decoded.unused_bits == 0
    printed = [
        _text(op.type.name, value)
        for op, value in zip(instruction.operands, decoded.operands, strict=True)
    ]
    assert _word(" ".join([instruction.mnemonic, *printed]) + ";;") == word


def test_slots_fill_in_source_order_nop_taking_the_first_free():
    (compound,) = AssemblySession("NOP; ADD LR0 LR1 LR2; SUB LR3 LR1 LR2;;").parse()
    placed = [(p.slot, p.instance) for p in encode.assign_slots(compound)]
    assert placed == [encode.SLOT_INSTANCES[0], (Slot.LR, 0), (Slot.LR, 1)]


def test_a_reserved_code_assembles_for_the_hardware_to_fault_on():
    # acc_stride_enums: padding names like reserved3 assemble but fail to execute.
    word = _word("ACC.STRIDE 16 reserved3 off LR0;;")
    decoded = WORD_CODEC.unpack(word)[(Slot.ACC, 0)]
    assert decoded.instruction.mnemonic == "ACC.STRIDE"
    assert decoded.operands[1] == 3
    assert decoded.unused_bits == 0


def test_an_empty_word_is_every_slot_nop():
    assert _word("NOP;;") == WORD_CODEC.pack({})


def test_slot_errors_keep_the_assembler_class_names():
    # The editor matches these messages (vscode-ipu-asm editing.js).
    with pytest.raises(SlotError, match=r"Too many instructions of type AccInst \(max 1\)"):
        _word("ACC.ADD; ACC.MAX;;")
    with pytest.raises(SlotError, match="Opcode 'MAC.EE' not found in any Inst subclass."):
        _word("MAC.EE R0;;")
    assert {slot.value: name for slot, name in encode.LEGACY_SLOT_NAME.items()} == {
        "break": "BreakInst",
        "load": "LoadInst",
        "store": "StoreInst",
        "acc_store": "AccStoreInst",
        "mult": "MultInst",
        "acc": "AccInst",
        "aaq": "AaqInst",
        "lr": "LrInst",
        "cond": "CondInst",
    }


def test_operand_errors_are_checked_before_the_next_instruction_is_placed():
    # The first instruction's bad operand count is reported, not the second
    # instruction's slot overflow.
    with pytest.raises(OperandError, match="expects"):
        _word("ACC.ADD R0; ACC.MAX;;")


def test_a_simulation_only_slot_warns():
    with pytest.warns(UserWarning, match="simulation-only acc_store slot"):
        AssemblySession("STR_ACC_REG LR0 CR0;;").assemble()


def test_disassembling_fields_is_disassembling_the_word():
    word = _word("ADD LR3 LR1 LR2; ACC.ADD;;")
    fields = DecodedWord.from_word(word)
    assert encode.disassemble_fields(fields) == encode.disassemble_word(word)
    # Only the non-zero fields: the rest read as 0, as in the packed word.
    partial = {name: value for name, value in fields.items() if value}
    assert encode.disassemble_fields(partial) == encode.disassemble_word(word)
    assert encode.disassemble_fields({}) == encode.disassemble_word(0)


def test_operation_text_is_assembly_order_without_unused_fields():
    word = _word("SET LR0 CR1;;")
    decoded = WORD_CODEC.unpack(word)[(Slot.LR, 0)]
    assert encode.operation_text(decoded) == "SET lr0 cr1"
    assert encode.operation_text(WORD_CODEC.unpack(word)[(Slot.ACC, 0)]) == "NOP"
