"""The ISA's word codec reproduces every assembled program of the corpus (E03-16).

For each program of the E02-01 corpus: decoding the `.bin` with
``unpack_program`` and re-encoding gives the same words, and ``pack_program``
of those words gives the same bytes the assembler wrote.
"""

from __future__ import annotations

from pathlib import Path

import asm_corpus
import pytest

from ipu_isa.codec.word import DecodedWord, pack_program, unpack_program

ROOT = Path.cwd()
ENTRIES = asm_corpus.discover(ROOT)


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda entry: entry.key)
def test_codec_round_trips_the_program(entry):
    program = asm_corpus.assemble(ROOT, entry)
    words = program.words
    decoded = unpack_program(program.bin_bytes)
    assert [d.to_word() for d in decoded] == words
    assert [DecodedWord.from_word(w) for w in words] == decoded
    assert pack_program(words) == program.bin_bytes
