"""``docs/content/requirements/faults.md`` agrees with ``ipu_isa.faults`` (E03-15).

The requirements page stays hand-written. These tests check that its fault-list
table states the same IDs, illegal states and detectors as :data:`FAULTS`, and
that its F-1 table is the one ``faults_table.f1_table`` derives from the ISA.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path

import faults_table
from faults_table import count_word, f1_table, table_span, with_table
from one_edit import view_with_dummy

from ipu_isa.faults import (
    FAULTS,
    Detector,
    FaultId,
    FaultInfo,
    reserved_encodings,
)
from ipu_isa.slots import Slot
from ipu_isa.view import isa

# Bazel runs the test from the runfiles root of the main repository.
FAULTS_PAGE = Path.cwd() / faults_table.FAULTS_PAGE

_DETECTOR_TEXT = {"CTRL": Detector.CTRL, "Cache agent": Detector.CACHE_AGENT}


def _table_after(text: str, marker: str) -> list[list[str]]:
    """The body rows of the first Markdown table after the line ``marker``."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == marker)
    rows: list[list[str]] = []
    in_table = False
    for line in lines[start + 1 :]:
        if line.startswith("|"):
            in_table = True
            rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
        elif in_table:
            break
    return rows[2:]  # drop the header and the --- row


def _page() -> str:
    return FAULTS_PAGE.read_text(encoding="utf-8")


def fault_table_mismatches(faults: Mapping[FaultId, FaultInfo], rows: list[list[str]]) -> list[str]:
    """How the fault-list table ``rows`` differs from ``faults``; empty when equal."""
    found = {row[0]: (row[1], _DETECTOR_TEXT.get(row[2])) for row in rows}
    wanted = {info.id.label: (info.summary, info.detected_in) for info in faults.values()}
    problems = [f"{label}: missing from faults.md" for label in wanted.keys() - found.keys()]
    problems += [f"{label}: not in FAULTS" for label in found.keys() - wanted.keys()]
    for label in sorted(wanted.keys() & found.keys()):
        if found[label] != wanted[label]:
            problems.append(f"{label}: faults.md says {found[label]}, FAULTS says {wanted[label]}")
    return problems


def test_fault_list_table_equals_faults():
    rows = _table_after(_page(), "## Fault list")
    assert fault_table_mismatches(FAULTS, rows) == []


def test_a_wording_change_in_faults_is_caught():
    rows = _table_after(_page(), "## Fault list")
    planted = dict(FAULTS)
    planted[FaultId.F3] = dataclasses.replace(
        FAULTS[FaultId.F3], summary="Two LR lanes of one word write one LR."
    )
    assert fault_table_mismatches(planted, rows) == [
        "F-3: faults.md says ('Two LR lanes of one word write the same LR.', "
        "<Detector.CTRL: 'ctrl'>), FAULTS says ('Two LR lanes of one word write one LR.', "
        "<Detector.CTRL: 'ctrl'>)"
    ]


def test_f1_table_is_the_derived_table():
    """The page holds ``f1_table`` exactly; ``tools/update_baselines.sh`` rewrites it."""
    start, end = table_span(_page())
    assert _page()[start:end] == f1_table(isa()), "run tools/update_baselines.sh"


def test_an_extra_instruction_changes_the_f1_table():
    changed = view_with_dummy()  # one more LR instruction (E03-24)
    count = len(reserved_encodings(isa())[Slot.LR].opcodes)
    new_row = next(r for r in f1_table(changed).splitlines() if r.startswith("| `LR`"))
    assert new_row.startswith(f"| `LR` | {count_word(count - 1).capitalize()} opcode")
    assert with_table(_page(), changed) != _page()
    assert with_table(with_table(_page(), changed), isa()) == _page()
