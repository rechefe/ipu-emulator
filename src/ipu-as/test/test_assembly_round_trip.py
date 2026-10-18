"""Disassembling to assembly syntax and assembling again gives the same words (E05-15).

For each program of the E02-01 corpus, for the emulator word and the hardware
word: ``program_text`` of the assembled words, assembled again, is the same
words. A program with an emulator-only instruction has no hardware words and
is checked for the emulator only.
"""

from __future__ import annotations

from pathlib import Path

import asm_corpus
import pytest
from program_checks import check_disassembly_round_trip

ROOT = Path.cwd()
ENTRIES = asm_corpus.discover(ROOT)


@pytest.mark.parametrize("target", ["emulator", "hardware"])
@pytest.mark.parametrize("entry", ENTRIES, ids=lambda entry: entry.key)
def test_the_assembly_text_assembles_to_the_same_words(entry, target):
    text = (ROOT / entry.path).read_text()
    if not check_disassembly_round_trip(text, target, entry.context):
        pytest.skip("an emulator-only instruction: no hardware words")
