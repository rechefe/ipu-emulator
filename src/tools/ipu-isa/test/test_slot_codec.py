"""The slot and word codecs (E03-17)."""

from __future__ import annotations

import dataclasses

import pytest

from ipu_isa.codec.slot import (
    OPCODE_SLOT,
    SLOT_CODECS,
    WORD_CODEC,
    EncodeError,
    ReservedEncoding,
    codecs,
)
from ipu_isa.codec.word import DecodedWord, opcode_field
from ipu_isa.operand_types import OPERAND_TYPES
from ipu_isa.slots import SLOTS, Slot
from ipu_isa.view import authoring_copies, isa, load_isa

CASES = [(slot, i) for slot in Slot for i in isa().instructions(slot)]


def _ends(operand) -> list[int]:
    """The lowest and highest encoding of an operand."""
    bits = isa().layout.operand_bits[operand.type.name]
    count = len(operand.type.domain.members) or 1 << bits
    return [0, count - 1]


@pytest.mark.parametrize("slot, instruction", CASES, ids=lambda x: str(getattr(x, "mnemonic", x)))
def test_encode_decode_round_trip_at_both_ends(slot, instruction):
    codec = SLOT_CODECS[slot]
    for end in (0, 1):
        operands = [_ends(op)[end] for op in instruction.operands]
        value = codec.encode(instruction.mnemonic, operands)
        assert value < 1 << codec.bits
        decoded = codec.decode(value)
        assert decoded.instruction is instruction
        assert decoded.operands == tuple(operands)
        assert decoded.unused_bits == 0


def test_mnemonics_are_case_insensitive():
    assert SLOT_CODECS[Slot.LR].encode("add", [1, 2, 3]) == SLOT_CODECS[Slot.LR].encode(
        "ADD", [1, 2, 3]
    )


def test_encode_errors_name_the_problem():
    lr = SLOT_CODECS[Slot.LR]
    with pytest.raises(EncodeError, match="no instruction 'NOPE'"):
        lr.encode("NOPE", [])
    with pytest.raises(EncodeError, match="takes 3 operand"):
        lr.encode("ADD", [1, 2])
    with pytest.raises(EncodeError, match="operand dest = 16"):
        lr.encode("ADD", [16, 0, 0])
    with pytest.raises(EncodeError, match="operand dest = -1"):
        lr.encode("ADD", [-1, 0, 0])


def test_nop_is_the_opcode_alone():
    for slot, codec in SLOT_CODECS.items():
        nop = isa().instruction(slot, "NOP")
        assert codec.nop() == nop.opcode << (codec.bits - isa().layout.unions[slot].opcode_bits)


def test_unused_opcodes_are_reserved():
    for slot, codec in SLOT_CODECS.items():
        used = len(isa().instructions(slot))
        opcode_bits = isa().layout.unions[slot].opcode_bits
        if used == 1 << opcode_bits:
            continue
        with pytest.raises(ReservedEncoding, match=f"opcode {used} is reserved"):
            codec.decode(used << (codec.bits - opcode_bits))


def test_unused_bits_are_reported():
    lr = SLOT_CODECS[Slot.LR]
    nop = lr.nop()
    assert lr.decode(nop | 1).unused_bits == 1


def test_word_pack_and_unpack():
    word = WORD_CODEC.pack({(Slot.LR, 1): SLOT_CODECS[Slot.LR].encode("ADD", [1, 2, 3])})
    slots = WORD_CODEC.unpack(word)
    assert slots[(Slot.LR, 1)].instruction.mnemonic == "ADD"
    assert slots[(Slot.LR, 1)].operands == (1, 2, 3)
    assert all(d.instruction.mnemonic == "NOP" for k, d in slots.items() if k != (Slot.LR, 1))
    assert WORD_CODEC.pack({k: d_value for k, d_value in _values(word).items()}) == word
    decoded = DecodedWord.from_word(word)
    assert decoded[opcode_field(Slot.LR, 1)] == isa().instruction(Slot.LR, "ADD").opcode
    assert set(slots) == {(s, i) for s in Slot for i in range(SLOTS[s].count)}


def _values(word: int) -> dict:
    return {
        key: SLOT_CODECS[key[0]].encode(d.instruction.mnemonic, d.operands)
        for key, d in WORD_CODEC.unpack(word).items()
    }


def test_word_pack_rejects_unknown_instances_and_wide_values():
    with pytest.raises(EncodeError, match="no such slot instance"):
        WORD_CODEC.pack({(Slot.MULT, 1): 0})
    with pytest.raises(EncodeError, match="wider than"):
        WORD_CODEC.pack({(Slot.MULT, 0): 1 << SLOT_CODECS[Slot.MULT].bits})


def test_opcode_slot():
    assert OPCODE_SLOT["ADD"] is Slot.LR
    assert OPCODE_SLOT["STR_ACC_REG"] is Slot.ACC_STORE
    assert "NOP" not in OPCODE_SLOT
    assert codecs() is codecs()


def test_a_modified_isa_gets_its_own_codecs():
    spec, pseudo = authoring_copies()
    entry = spec["lr"]["ADD"]
    entry["doc"] = dataclasses.replace(entry["doc"], example="TESTNEG LR0 LR1 LR2;;")
    spec["lr"]["TESTNEG"] = entry
    view = load_isa(spec, pseudo, SLOTS, OPERAND_TYPES)
    lr = codecs(view).slots[Slot.LR]
    value = lr.encode("TESTNEG", [1, 2, 3])
    assert lr.decode(value).instruction.mnemonic == "TESTNEG"
    assert codecs(view).opcode_slot["TESTNEG"] is Slot.LR
    with pytest.raises(EncodeError):
        SLOT_CODECS[Slot.LR].encode("TESTNEG", [1, 2, 3])
