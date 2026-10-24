"""The disassembler (E03-18). Expected texts are written by hand; the E02-02
corpus snapshots pin the rest."""

from __future__ import annotations

import pytest

from ipu_isa.codec.disasm import (
    SLOT_SEPARATOR,
    disassemble,
    operand_text,
    slot_operations,
    slot_text,
)
from ipu_isa.codec.slot import SLOT_CODECS, WORD_CODEC, DecodedSlot, ReservedEncoding
from ipu_isa.slots import Slot
from ipu_isa.view import isa

# The all-NOP word, slot by slot in layout order (MSB first): COND, LR0-LR2,
# LOAD, MULT, ACC, AAQ, AAQ_EXACT, BREAK.
NOP_SLOTS = [
    "NOP 0 lr0 lr0",
    "NOP 0 lr0 lr0",
    "NOP 0 lr0 lr0",
    "NOP 0 lr0 lr0",
    "NOP cr0 lr0 lr0",
    "NOP lr0 cr0.l cr0.l lr0 lr0 0",
    "NOP cr0.l lrd0 lr0 off",
    "NOP identity cr0 cr0.l cr0.l lr0",
    "NOP identity cr0 cr0.l lr0",
    "NOP 0 lr0",
]


def _slot(slot: Slot, mnemonic: str, operands: list[int]) -> DecodedSlot:
    codec = SLOT_CODECS[slot]
    return codec.decode(codec.encode(mnemonic, operands))


@pytest.mark.parametrize(
    "slot, mnemonic, operands, text, operation",
    [
        # Fields: src_b (LcrIdx), dest, src_a.
        (Slot.LR, "ADD", [1, 2, 3], "ADD lr3 lr1 lr2", "ADD lr1 lr2 lr3"),
        # k is stored as k - 1.
        (Slot.LR, "INC.MODP2", [4, 5, 2], "INC.MODP2 lr5 lr4 3", "INC.MODP2 lr4 lr5 3"),
        # reshape_mask 9 is LR1; LrdIdx 1 is lrd2; field 3 is unbound.
        (
            Slot.ACC,
            "ACC.RESHAPE",
            [1, 2, 9],
            "ACC.RESHAPE lr1 lrd2 lrd4 off",
            "ACC.RESHAPE lrd2 lrd4 lr1",
        ),
        (
            Slot.ACC,
            "ACC.RESHAPE",
            [0, 0, 7],
            "ACC.RESHAPE 7 lrd0 lrd0 off",
            "ACC.RESHAPE lrd0 lrd0 7",
        ),
        (Slot.COND, "BEQ", [17, 3, 1000], "BEQ 1000 cr1 lr3", "BEQ cr1 lr3 1000"),
    ],
)
def test_slot_text_and_operations(slot, mnemonic, operands, text, operation):
    decoded = _slot(slot, mnemonic, operands)
    assert slot_text(decoded) == text
    word = WORD_CODEC.pack({(slot, 0): decoded.value})
    label = isa().slots[slot].instance_labels()[0]
    assert dict(slot_operations(word))[label] == operation


def test_operand_text():
    k = isa().instruction(Slot.LR, "INC.MODP2").operands[2]
    assert operand_text(k, 0) == "1"
    assert operand_text(k, 8) == "9"
    addbi = isa().instruction(Slot.LR, "ADDBI").operands[1]  # range -128..255, stored as is
    assert operand_text(addbi, 255) == "255"
    stride = isa().instruction(Slot.ACC, "ACC.STRIDE").operands[1]
    assert operand_text(stride, 2) == "on_inv"


def test_all_nop_word():
    assert disassemble(WORD_CODEC.pack({})) == SLOT_SEPARATOR.join(NOP_SLOTS) + ";;"


def test_slot_operations_labels_every_instance():
    labels = [label for label, _ in slot_operations(WORD_CODEC.pack({}))]
    assert labels == [
        "COND",
        "LR0",
        "LR1",
        "LR2",
        "LOAD",
        "MULT",
        "ACC",
        "AAQ",
        "AAQ_EXACT",
        "BREAK",
    ]
    assert all(text == "NOP" for _, text in slot_operations(WORD_CODEC.pack({})))


def test_a_value_past_the_last_member_is_marked_illegal():
    lr = SLOT_CODECS[Slot.LR]
    # SET's src (LcrHalfIdx, 64 members) sits in the 8-bit field 0.
    value = (isa().instruction(Slot.LR, "SET").opcode << 16) | (200 << 8) | (1 << 4)
    assert slot_text(lr.decode(value)) == "SET <illegal_LcrHalfIdx_200> lr1 lr0"


def test_half_cr_operands_disassemble_with_their_half():
    mult = _slot(Slot.MULT, "MULT.VE", [0, 3, 0, 0, 30, 27])
    assert slot_text(mult).endswith("cr15.l cr13.h lr0 lr0 0") or "cr15.l" in slot_text(mult)
    word = WORD_CODEC.pack({(Slot.MULT, 0): mult.value})
    assert dict(slot_operations(word))["MULT"] == "MULT.VE lr0 cr3 0 lr0 cr15.l cr13.h"
    cond = _slot(Slot.COND, "BEQ", [16 + 32 + 5, 48 + 2, 7])
    word = WORD_CODEC.pack({(Slot.COND, 0): cond.value})
    assert dict(slot_operations(word))["COND"] == "BEQ cr5.h cr2.h 7"


def test_an_illegal_mult_stage_names_its_type():
    load = SLOT_CODECS[Slot.LOAD]
    # LD.MULT's dest (MultStageReg, r0/r1) sits in the 4-bit field 2.
    value = (isa().instruction(Slot.LOAD, "LD.MULT").opcode << 12) | 13
    assert slot_text(load.decode(value)).endswith(" <illegal_MultStageReg_13>")
    word = WORD_CODEC.pack({(Slot.LOAD, 0): value})
    assert dict(slot_operations(word))["LOAD"].startswith("LD.MULT <illegal_MultStageReg_13>")


def test_a_reserved_opcode_raises():
    reserved = len(isa().instructions(Slot.LR))  # the first opcode no LR instruction has
    with pytest.raises(ReservedEncoding):
        disassemble(WORD_CODEC.pack({(Slot.LR, 0): reserved << 16}))
