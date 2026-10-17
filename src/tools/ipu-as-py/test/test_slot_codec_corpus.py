"""The ISA's slot codec agrees with the assembler on every corpus word (E03-17).

For every word of the E02-01 corpus (assembled by ``ipu_as``):
``WORD_CODEC.unpack`` then ``pack`` gives the same word, and re-encoding each
decoded slot with ``SlotCodec.encode`` gives the bits the assembler's
``Inst.encode`` produced. The all-NOP word equals the assembler's ``NOP;;``.
"""

from __future__ import annotations

from pathlib import Path

import asm_corpus
import pytest
from program_checks import check_slot_codec

from ipu_as.lark_tree import assemble
from ipu_isa.codec.slot import WORD_CODEC

ROOT = Path.cwd()
ENTRIES = asm_corpus.discover(ROOT)


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda entry: entry.key)
def test_codec_matches_the_assembler(entry):
    check_slot_codec(asm_corpus.assemble(ROOT, entry).words)


def test_all_nop_word():
    assert WORD_CODEC.pack({}) == assemble("NOP;;")[0]
