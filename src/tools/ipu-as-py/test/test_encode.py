"""The assembler encodes through the ipu_isa codec (E05-02).

Every instruction of every slot round-trips: assembled alone, its slot decodes
(``SLOT_CODECS``) to the same instruction, and the operands printed back
assemble to the same word.
"""

from __future__ import annotations

import warnings

import pytest

from ipu_as import encode
from ipu_as.compound_inst import CompoundInst
from ipu_as.errors import OperandError, SlotError
from ipu_as.inst import OPERAND_TYPE_MAP
from ipu_as.session import AssemblySession
from ipu_isa.codec.slot import WORD_CODEC
from ipu_isa.slots import LAYOUT_ORDER
from ipu_isa.view import isa


def _word(source: str) -> int:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (word,) = AssemblySession(source).assemble().words
    return word


def _text(operand_type: str, value: int) -> str:
    text = OPERAND_TYPE_MAP[operand_type].decode(value)
    # A label decodes to its address; at address 0 that is the offset +N.
    return f"+{text}" if operand_type == "Label" else text


def _instructions():
    for slot in LAYOUT_ORDER:
        for instruction in isa().instructions(slot):
            if instruction.mnemonic != "NOP":
                yield pytest.param(slot, instruction, id=f"{slot.value}.{instruction.mnemonic}")


@pytest.mark.parametrize(("slot", "instruction"), _instructions())
def test_every_instruction_round_trips(slot, instruction):
    operands = [OPERAND_TYPE_MAP[op.type.name].default().token.value for op in instruction.operands]
    word = _word(" ".join([instruction.mnemonic, *operands]) + ";;")
    decoded = WORD_CODEC.unpack(word)[(slot, 0)]
    assert decoded.instruction.mnemonic == instruction.mnemonic
    printed = [
        _text(op.type.name, value)
        for op, value in zip(instruction.operands, decoded.operands, strict=True)
    ]
    assert _word(" ".join([instruction.mnemonic, *printed]) + ";;") == word
    assert instruction.mnemonic in CompoundInst.decode(word)


def test_an_empty_word_is_every_slot_nop():
    assert _word("NOP;;") == WORD_CODEC.pack({})


def test_slot_errors_keep_the_assembler_class_names():
    # The editor matches these messages (vscode-ipu-asm editing.js).
    with pytest.raises(SlotError, match=r"Too many instructions of type AccInst \(max 1\)"):
        _word("ACC.ADD; ACC.MAX;;")
    with pytest.raises(SlotError, match="Opcode 'MAC.EE' not found in any Inst subclass."):
        _word("MAC.EE R0;;")
    assert encode.LEGACY_SLOT_NAME[LAYOUT_ORDER[0]].endswith("Inst")


def test_operand_errors_are_checked_before_the_next_instruction_is_placed():
    # The first instruction's bad operand count is reported, not the second
    # instruction's slot overflow.
    with pytest.raises(OperandError, match="expects"):
        _word("ACC.ADD R0; ACC.MAX;;")


def test_a_simulation_only_slot_warns():
    with pytest.warns(UserWarning, match="simulation-only acc_store slot"):
        AssemblySession("STR_ACC_REG LR0 CR0;;").assemble()
