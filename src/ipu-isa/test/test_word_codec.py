"""The instruction-word codec (E03-16)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from ipu_isa.codec.word import (
    WORD_LAYOUT,
    DecodedWord,
    field_name,
    hardware_word,
    opcode_field,
    pack_program,
    unpack_program,
    unpack_words,
    word_bytes,
)
from ipu_isa.slots import Slot
from ipu_isa.tags import Tag
from ipu_isa.view import isa

#: E02-02's snapshot of the assembler's word fields, LSB first.
WORD_FIELDS = Path("src/ipu-as/test/baselines/word_fields.json")

_WORDS = [0, (1 << 209) - 1, *(random.Random(3).getrandbits(209) for _ in range(64))]


def test_layout_matches_the_assembler_snapshot():
    snapshot = json.loads(WORD_FIELDS.read_text())
    assert [[f.name, f.bits] for f in WORD_LAYOUT] == snapshot["fields_lsb_first"]
    assert sum(f.bits for f in WORD_LAYOUT) == snapshot["total_bits"] == 209


def test_fields_tile_the_word():
    shift = 0
    for f in WORD_LAYOUT:
        assert f.shift == shift
        shift += f.bits
    assert len({f.name for f in WORD_LAYOUT}) == len(WORD_LAYOUT)


def test_field_names():
    assert opcode_field(Slot.LR, 1) == "lr_inst_1_token_0_lr_inst_opcode"
    assert opcode_field(Slot.MULT) == "mult_inst_token_0_mult_inst_opcode"
    assert field_name(Slot.BREAK, 0, 1) == "break_inst_token_1_break_immediate_type"


@pytest.mark.parametrize("word", _WORDS, ids=lambda word: f"{word:#x}"[:12])
def test_decode_encode_round_trip(word):
    decoded = DecodedWord.from_word(word)
    assert decoded.to_word() == word
    assert set(decoded) == {f.name for f in WORD_LAYOUT}


def test_missing_fields_encode_as_zero():
    assert DecodedWord().to_word() == 0
    assert DecodedWord({opcode_field(Slot.BREAK): 1}).to_word() == 1 << 20


def test_slot_value():
    word = DecodedWord.from_word(_WORDS[2])
    for slot in Slot:
        for instance in range(isa().slots[slot].count):
            fields = [f for f in WORD_LAYOUT if f.slot is slot and f.instance == instance]
            low, high = fields[0].shift, fields[-1].shift + fields[-1].bits
            expected = (_WORDS[2] >> low) & ((1 << (high - low)) - 1)
            assert word.slot_value(slot, instance) == expected


def test_word_bytes():
    assert word_bytes() == 28
    assert word_bytes(Tag.HARDWARE) == 24


def test_hardware_word_drops_the_emulator_slots():
    emulator = [f for f in WORD_LAYOUT if isa().slots[f.slot].tag is Tag.EMULATOR]
    assert {f.slot for f in emulator} == {Slot.AAQ_EXACT, Slot.BREAK}
    low_bits = 209 - 169  # emulator word minus hardware word (E03-07)
    assert sum(f.bits for f in emulator) == low_bits
    assert all(f.shift < low_bits for f in emulator)
    for word in _WORDS:
        assert hardware_word(word) == word >> low_bits
        assert hardware_word(word).bit_length() <= isa().layout.word_bits(Tag.HARDWARE)


def test_unpack_words():
    assert unpack_words(bytes(range(28)) + b"\xff") == [int.from_bytes(bytes(range(28)), "little")]


def test_program_round_trip():
    data = pack_program(_WORDS)
    assert len(data) == len(_WORDS) * 28
    assert [w.to_word() for w in unpack_program(data)] == _WORDS
    assert unpack_program(data + b"\x00" * 5) == unpack_program(data)  # partial word ignored
