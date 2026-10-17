"""Disassembling to assembly syntax and assembling again gives the same words (E05-15).

For each program of the E02-01 corpus, for the emulator word and the hardware
word: ``program_text`` of the assembled words, assembled again, is the same
words. A program with an emulator-only instruction has no hardware words and
is checked for the emulator only.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import asm_corpus
import pytest

from ipu_as.errors import SlotError
from ipu_as.session import AssemblySession
from ipu_isa.codec.disasm import program_text

ROOT = Path.cwd()
ENTRIES = asm_corpus.discover(ROOT)


def _session(entry: asm_corpus.CorpusEntry) -> AssemblySession:
    text = (ROOT / entry.path).read_text()
    if entry.context is None:
        return AssemblySession(text)
    return AssemblySession(text, defines=entry.context)


@pytest.mark.parametrize("target", ["emulator", "hardware"])
@pytest.mark.parametrize("entry", ENTRIES, ids=lambda entry: entry.key)
def test_the_assembly_text_assembles_to_the_same_words(entry, target):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # emulator-only slots in the emulator word
        try:
            words = _session(entry).assemble(target).words
        except SlotError:
            assert target == "hardware"
            pytest.skip("an emulator-only instruction: no hardware words")
        source = program_text(words, target=target)
        assert AssemblySession(source).assemble(target).words == words
