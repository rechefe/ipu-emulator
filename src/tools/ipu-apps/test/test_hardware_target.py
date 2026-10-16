"""Every E02 corpus program assembles for the hardware unless it uses an
emulator-only instruction, and its hardware words are the top bits of its
emulator words (E05-15)."""

from __future__ import annotations

import warnings
from pathlib import Path

import asm_corpus
import pytest

from ipu_as.errors import SlotError
from ipu_as.session import AssemblySession
from ipu_isa.codec.slot import WORD_CODEC
from ipu_isa.codec.word import hardware_word
from ipu_isa.slots import SLOTS
from ipu_isa.tags import Tag

ROOT = Path.cwd()
ENTRIES = asm_corpus.discover(ROOT)


def _uses_emulator_slots(word: int) -> bool:
    return any(
        SLOTS[slot].tag is Tag.EMULATOR and decoded.instruction.mnemonic != "NOP"
        for (slot, _), decoded in WORD_CODEC.unpack(word).items()
    )


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda entry: entry.key)
def test_hardware_words_are_the_emulator_words_top_bits(entry):
    text = (ROOT / entry.path).read_text()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emulator = AssemblySession(text, defines=entry.context).assemble().words
    try:
        hardware = AssemblySession(text, defines=entry.context).assemble("hardware").words
    except SlotError as error:
        assert "emulator-only" in str(error)
        assert any(_uses_emulator_slots(word) for word in emulator)
        return
    assert not any(_uses_emulator_slots(word) for word in emulator)
    assert hardware == tuple(hardware_word(word) for word in emulator)


def test_some_corpus_programs_are_hardware_programs():
    hardware = 0
    for entry in ENTRIES:
        text = (ROOT / entry.path).read_text()
        try:
            AssemblySession(text, defines=entry.context).assemble("hardware")
            hardware += 1
        except SlotError:
            pass
    assert hardware > 10
