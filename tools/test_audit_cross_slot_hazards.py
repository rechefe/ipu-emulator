"""The cross-slot hazard audit (E04-25): its word decoding and the corpus result."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from ipu_as.session import assemble
from ipu_isa.timing import View

sys.path.insert(0, str(Path(__file__).parent))

import audit_cross_slot_hazards as audit  # noqa: E402

# Bazel runs the test from the runfiles root, where the corpus sits at its
# repo-relative paths.
ROOT = Path.cwd()


def _hazards(source: str) -> list[audit.Hazard]:
    (word,) = assemble(source).words
    return list(audit.word_hazards("probe", 0, word))


def test_agg_reading_an_lr_the_word_writes_is_a_hazard():
    (hazard,) = _hazards("SET LR3 CR2 ; AGG.SUM LR3 CR15 ;;")
    assert (hazard.reader, hazard.register, hazard.via_operand, hazard.indices) == (
        "acc.AGG.SUM",
        "lr",
        "dest_slot",
        (3,),
    )
    assert hazard.writers == ("lr[0].SET",)
    assert (hazard.declared, hazard.contract, hazard.deviates) == (
        View.FORWARDED,
        View.FORWARDED,
        False,
    )


def test_a_different_lr_is_no_hazard():
    assert _hazards("SET LR1 CR2 ; AGG.SUM LR3 CR15 ;;") == []


@pytest.mark.parametrize(("written", "operand"), [("LR1", "source"), ("LR4", "dest")])
def test_an_lrd_pair_overlaps_either_of_its_lrs(written, operand):
    (hazard,) = _hazards(f"SET {written} CR2 ; ACC.RESHAPE LRD0 LRD4 7 ;;")
    assert hazard.via_operand == operand
    assert hazard.indices == ((0, 1) if operand == "source" else (4, 5))


def test_a_listed_deviation_is_reported_with_its_issue():
    hazards = _hazards("MULT.RC.VV LR0 R0 0 LR2 CR15 ; ACC.RESHAPE LRD0 LRD2 0 ;;")
    (deviating,) = [h for h in hazards if h.deviates]
    assert (deviating.reader, deviating.register, deviating.fixed_by) == (
        "acc.ACC.RESHAPE",
        "mult_res",
        "E04-26",
    )
    assert (deviating.declared, deviating.contract) == (View.CYCLE_START, View.FORWARDED)
    assert "1 hazard(s) where the declared view deviates in 1 bundle(s)" in audit.report(
        hazards, show_all=False
    )


def test_no_program_relies_on_a_listed_deviation():
    """Every bundle of the corpus gives the same result under the declared view
    and the contract, so removing a KNOWN_DEVIATIONS row changes no program. A
    failure names the bundles to fix before the flip."""
    deviating = [h for h in audit.corpus_hazards(ROOT) if h.deviates]
    assert deviating == [], audit.report(deviating, show_all=False)
