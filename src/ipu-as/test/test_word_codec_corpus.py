"""The ISA's word codec reproduces every assembled program of the corpus (E03-16).

For each program of the E02-01 corpus: decoding the `.bin` with
``unpack_program`` and re-encoding gives the same words, and ``pack_program``
of those words gives the same bytes the assembler wrote. The assembler now
writes `.bin` through ``pack_program`` too, so the independent proof that the
bytes did not change is the E02-01 digest test (``test_asm_digests``).
"""

from __future__ import annotations

from pathlib import Path

import asm_corpus
import pytest
from program_checks import check_word_codec

ROOT = Path.cwd()
ENTRIES = asm_corpus.discover(ROOT)


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda entry: entry.key)
def test_codec_round_trips_the_program(entry):
    program = asm_corpus.assemble(ROOT, entry)
    check_word_codec(program.words, program.bin_bytes)
