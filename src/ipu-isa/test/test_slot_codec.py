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
    field_code,
)
from ipu_isa.codec.word import DecodedWord, opcode_field
from ipu_isa.faults import reserved_encodings
from ipu_isa.operand_types import OPERAND_TYPES
from ipu_isa.slots import SLOTS, Slot
from ipu_isa.view import authoring_copies, isa, load_isa

CASES = [(slot, i) for slot in Slot for i in isa().instructions(slot)]


# Which codes mean nothing comes from F-1's table (ipu_isa.faults); the hand-written
# test_reserved_operand_codes_are_rejected pins that table's answers for two types.
_RESERVED = {
    (slot, operand_type.operand_type): set(operand_type.codes)
    for slot, entry in reserved_encodings(isa()).items()
    for operand_type in entry.operands
}


def _ends(slot, operand) -> list[int]:
    """The lowest and highest meaningful encoding of an operand."""
    bits = isa().layout.operand_bits[operand.type.name]
    reserved = _RESERVED.get((slot, operand.type.name), set())
    meaningful = [code for code in range(1 << bits) if code not in reserved]
    return [meaningful[0], meaningful[-1]]


@pytest.mark.parametrize("slot, instruction", CASES, ids=lambda x: str(getattr(x, "mnemonic", x)))
def test_encode_decode_round_trip_at_both_ends(slot, instruction):
    codec = SLOT_CODECS[slot]
    for end in (0, 1):
        operands = [_ends(slot, op)[end] for op in instruction.operands]
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


def test_reserved_operand_codes_are_rejected():
    lr = SLOT_CODECS[Slot.LR]
    # k is stored as k - 1 and ranges over 1..9: codes 0..8 mean something, 9..15 do not.
    assert lr.decode(lr.encode("INC.MODP2", [0, 0, 8])).operands == (0, 0, 8)
    with pytest.raises(EncodeError, match="operand k = 9 is a reserved code"):
        lr.encode("INC.MODP2", [0, 0, 9])
    with pytest.raises(EncodeError, match="operand k = 15 is a reserved code"):
        lr.encode("INC.MODP2", [0, 0, 15])
    # HorizontalStride: off, on, on_inv, reserved3.
    acc = SLOT_CODECS[Slot.ACC]
    acc.encode("ACC.STRIDE", [0, 2, 0, 0])
    with pytest.raises(EncodeError, match="operand horizontal_stride = 3 is a reserved code"):
        acc.encode("ACC.STRIDE", [0, 3, 0, 0])


def test_encode_with_reserved_writes_a_reserved_code_where_encode_puts_it():
    lr = SLOT_CODECS[Slot.LR]
    base = lr.encode("INC.MODP2", [0, 0, 0])
    shift = (lr.encode("INC.MODP2", [0, 0, 1]) ^ base).bit_length() - 1  # where k's field starts
    assert lr.encode_with_reserved("INC.MODP2", [0, 0, 15]) == base | 15 << shift
    assert lr.encode_with_reserved("INC.MODP2", [0, 0, 8]) == lr.encode("INC.MODP2", [0, 0, 8])
    with pytest.raises(EncodeError, match="takes 3 operand"):
        lr.encode_with_reserved("INC.MODP2", [0, 0])
    with pytest.raises(EncodeError, match="outside"):
        lr.encode_with_reserved("INC.MODP2", [0, 0, 16])


def test_field_code_is_twos_complement_for_a_negative_value():
    assert [field_code(v, 8) for v in (0, 5, 255, -1, -128)] == [0, 5, 255, 255, 128]


def test_decode_rejects_values_outside_the_slot():
    lr = SLOT_CODECS[Slot.LR]
    with pytest.raises(EncodeError, match="value -1 does not fit 20 bits"):
        lr.decode(-1)
    with pytest.raises(EncodeError, match="does not fit 20 bits"):
        lr.decode(1 << 20)


# NOP's slot value, by hand from instruction_spec.py: its opcode in the top bits.
NOP_VALUES = {
    Slot.BREAK: 0x200000,  # 22 bits, 2-bit opcode 2
    Slot.LOAD: 0x3000,  # 14 bits, 2-bit opcode 3
    Slot.MULT: 0xC000000,  # 29 bits, 3-bit opcode 3
    Slot.ACC: 0x18000,  # 18 bits, 4-bit opcode 6
    Slot.AAQ: 0x0,  # 23 bits, 2-bit opcode 0
    Slot.AAQ_EXACT: 0x0,  # 18 bits, 1-bit opcode 0
    Slot.LR: 0x80000,  # 20 bits, 4-bit opcode 8
    Slot.COND: 0x1800000,  # 25 bits, 3-bit opcode 6
}


def test_nop_is_the_opcode_alone():
    assert {slot: codec.nop() for slot, codec in SLOT_CODECS.items()} == NOP_VALUES


def test_unused_opcodes_are_reserved():
    for slot, codec in SLOT_CODECS.items():
        opcode_bits = isa().layout.unions[slot].opcode_bits
        reserved = tuple(range(len(isa().instructions(slot)), 1 << opcode_bits))
        for opcode in reserved[:1] + reserved[-1:]:
            with pytest.raises(ReservedEncoding, match=f"opcode {opcode} is reserved"):
                codec.decode(opcode << (codec.bits - opcode_bits))
    assert ReservedEncoding.fault == 1


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
    assert WORD_CODEC.pack(_values(word)) == word
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
    with pytest.raises(EncodeError, match="does not fit 29 bits"):
        WORD_CODEC.pack({(Slot.MULT, 0): 1 << 29})
    with pytest.raises(EncodeError, match="value -1 does not fit"):
        WORD_CODEC.pack({(Slot.MULT, 0): -1})


def test_opcode_slot():
    assert OPCODE_SLOT["ADD"] is Slot.LR
    assert OPCODE_SLOT["ACTIVATE.EXACT"] is Slot.AAQ_EXACT
    assert OPCODE_SLOT["LD.LUT"] is Slot.AAQ
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
