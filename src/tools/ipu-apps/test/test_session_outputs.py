"""AssemblyResult.to_bin()/to_mem() equal the files the assembler's file entry points write.

Checked for every E02 corpus program (E05-05). Each session is fresh, so the
in-process cache does not hide a difference.
"""

from __future__ import annotations

from pathlib import Path

import asm_corpus
import jinja2
import pytest
from ipu_as.session import AssemblySession

ROOT = Path.cwd()
ENTRIES = asm_corpus.discover(ROOT)


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda entry: entry.key)
def test_result_bytes_equal_the_written_files(entry):
    text = (ROOT / entry.path).read_text()
    if entry.context is not None:
        text = jinja2.Template(text).render(**entry.context)
    result = AssemblySession(text).assemble()
    written = asm_corpus.assemble(ROOT, entry)
    assert result.to_bin() == written.bin_bytes
    assert result.to_mem().encode() == written.mem_bytes
