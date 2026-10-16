"""``docs/content/requirements/faults.md`` agrees with ``ipu_isa.faults`` (E03-15).

The requirements page stays hand-written. These tests check that its fault-list
table states the same IDs, illegal states and detectors as :data:`FAULTS`, and
that its F-1 table states the reserved encodings :func:`reserved_encodings`
derives from the ISA.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path

from ipu_isa.faults import (
    FAULTS,
    Detector,
    FaultId,
    FaultInfo,
    ReservedEncodings,
    ReservedOperand,
    reserved_encodings,
)
from ipu_isa.operand_types import OperandKind
from ipu_isa.slots import Slot
from ipu_isa.view import isa

# Bazel runs the test from the runfiles root of the main repository.
FAULTS_PAGE = Path.cwd() / "docs/content/requirements/faults.md"

_DETECTOR_TEXT = {"CTRL": Detector.CTRL, "Cache agent": Detector.CACHE_AGENT}
_NUMBER_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight")


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


def _opcode_phrase(entry: ReservedEncodings) -> str:
    count = len(entry.opcodes)
    noun = "opcode" if count == 1 else "opcodes"
    return f"{_NUMBER_WORDS[count]} {noun} of the {entry.opcode_bits}-bit field"


def _operand_phrases(reserved: ReservedOperand) -> list[str]:
    """Text the F-1 table must contain for ``reserved``."""
    operand_type = isa().operand_types[reserved.operand_type]
    members = operand_type.domain.members
    if operand_type.kind is OperandKind.REGISTER:
        valid = [f"`{m.upper()}`" for code, m in enumerate(members) if code not in reserved.codes]
        return [f"`{operand_type.name}` values other than " + " and ".join(valid)]
    if operand_type.name == "ActivationFn":
        return ["Activation function codes with no named function"]
    if operand_type.kind is OperandKind.ENUM:
        return [f"above {min(reserved.codes) - 1}"] + [f"`{n}`" for n in reserved.operands]
    assert operand_type.domain.value_range is not None
    low, high = operand_type.domain.value_range
    return [f"`{name}` outside {low} to {high}" for name in reserved.operands]


def test_f1_table_states_the_derived_reserved_encodings():
    rows = _table_after(_page(), "Reserved encodings for F-1, from the instruction reference:")
    table = {Slot(row[0].strip("`").lower()): row[1] for row in rows}
    reserved = reserved_encodings(isa())
    assert set(table) == {slot for slot, entry in reserved.items() if not entry.empty}
    for slot, text in table.items():
        entry = reserved[slot]
        if entry.opcodes:
            assert _opcode_phrase(entry) in text.lower(), (slot, text)
        else:
            assert "opcode" not in text.lower(), (slot, text)
        for operand in entry.operands:
            for phrase in _operand_phrases(operand):
                assert phrase in text, (slot, phrase, text)
