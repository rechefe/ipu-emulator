"""The word exclusion table (ADR 0007)."""

from __future__ import annotations

from ipu_isa.exclusions import WORD_EXCLUSIONS, word_conflicts
from ipu_isa.slots import Slot
from ipu_isa.view import isa


def test_ld_lut_excludes_every_load_but_nop():
    for load in ("LD.MULT", "LD.MULT.CYCLIC", "LD.MASK"):
        (row,) = word_conflicts([(Slot.AAQ, "LD.LUT"), (Slot.LOAD, load)])
        assert "read port" in row.reason
    assert word_conflicts([(Slot.AAQ, "LD.LUT"), (Slot.LOAD, "NOP")]) == ()


def test_the_two_activations_exclude_each_other():
    (row,) = word_conflicts([(Slot.AAQ, "activate.quantize"), (Slot.AAQ_EXACT, "ACTIVATE.EXACT")])
    assert row is WORD_EXCLUSIONS[1]
    assert word_conflicts([(Slot.AAQ, "ACTIVATE.QUANTIZE"), (Slot.LOAD, "LD.MULT")]) == ()


def test_every_side_names_an_instruction_of_its_slot():
    import fnmatch

    for row in WORD_EXCLUSIONS:
        for slot, pattern in (row.first, row.second):
            names = [i.mnemonic for i in isa().instructions(slot)]
            assert fnmatch.filter(names, pattern), (slot, pattern)
