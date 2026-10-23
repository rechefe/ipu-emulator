"""Every E02 corpus program assembles for the hardware unless it uses an
emulator-only instruction, and its hardware words are the top bits of its
emulator words (E05-15)."""

from __future__ import annotations

from pathlib import Path

import asm_corpus
import pytest
from program_checks import assembles_for_hardware, check_hardware_target

ROOT = Path.cwd()
ENTRIES = asm_corpus.discover(ROOT)


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda entry: entry.key)
def test_hardware_words_are_the_emulator_words_top_bits(entry):
    check_hardware_target((ROOT / entry.path).read_text(), entry.context)


def test_some_corpus_programs_are_hardware_programs():
    hardware = sum(
        assembles_for_hardware((ROOT / entry.path).read_text(), entry.context) for entry in ENTRIES
    )
    # The editor's corpus; the kernel tree's count is
    # //src/ipu-harness:test_kernel_programs.
    assert hardware >= 1
