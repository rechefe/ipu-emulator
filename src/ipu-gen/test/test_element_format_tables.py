"""The element-format tables follow ``ipu_isa.formats`` (E09-19).

``explanation/element-formats.md`` includes its tables from the generated file, so a
format added to the enum gets its rows with no doc edit. The CR scalar operands of the
instruction reference link to the page's CR scalar section.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ipu_gen import docs_pages
from ipu_isa.formats import FORMATS, ElementFormat
from ipu_isa.view import isa

# Bazel runs the test from the runfiles root of the main repository.
PAGE = Path.cwd() / "docs/content/explanation/element-formats.md"
CR_SCALAR_LINK = "(../../explanation/element-formats.md#cr-scalars)"
_OPERAND_DOC = re.compile(r"`(?P<name>\w+)`")  # "`cr_idx`: ..." names its operand first

_SECTION = re.compile(
    r"<!-- --8<-- \[start:(?P<name>[\w-]+)\] -->\n(?P<body>.*?)<!-- --8<-- \[end:", re.S
)


def _sections(formats=FORMATS) -> dict[str, list[str]]:
    """Section name -> its table's body rows (header and --- row dropped)."""
    text = docs_pages.render_element_format_tables(formats)
    return {m["name"]: m["body"].splitlines()[2:] for m in _SECTION.finditer(text)}


def _named(rows: list[str]) -> list[str]:
    return [row.split("|")[1].strip().strip("`") for row in rows]


def test_one_row_per_format():
    sections = _sections()
    assert tuple(sections) == docs_pages.ELEMENT_FORMAT_SECTIONS
    every = [fmt.name for fmt in FORMATS]
    assert _named(sections["formats"]) == every
    assert _named(sections["cr-scalars"]) == every
    # One row per signed 8-bit NX8 type: QF exp_format 1-7.
    assert _named(sections["encodings"]) == [f"E{fe}" for fe in range(1, 8)]


def test_the_rows_follow_the_formats_given():
    kept = {ElementFormat.FP32: FORMATS[ElementFormat.FP32]}
    sections = _sections(kept)
    assert _named(sections["formats"]) == ["FP32"]
    assert _named(sections["encodings"]) == []  # no NX8: no NX8 types
    assert _named(sections["cr-scalars"]) == ["FP32"]


@pytest.mark.parametrize(
    ("row"),
    [
        # Hand-computed at scale 127 from ARCH-stage-mult.md §3.2.1: an exponent
        # below the clamp e (all ones) is (1 + m / 2^fm) * 2^-e; the clamp
        # exponent is m / 2^fm * 2^-(clamp - 1), a subnormal.
        "| `E4` | 4 | 3 | 1.875 | 2<sup>-14</sup> | 2<sup>-17</sup> |",
        "| `E1` | 1 | 6 | 1.984375 | 1 | 0.015625 |",
        "| `E7` | 7 | 0 | 1 | 2<sup>-126</sup> | 2<sup>-126</sup> |",
    ],
)
def test_encoding_rows(row):
    assert row in _sections()["encodings"]


def test_the_page_includes_every_section():
    page = PAGE.read_text(encoding="utf-8")
    for name in docs_pages.ELEMENT_FORMAT_SECTIONS:
        include = f'--8<-- "docs/generated/{docs_pages.ELEMENT_FORMAT_TABLES}:{name}"'
        assert include in page, name


def test_cr_scalar_operands_link_to_the_page():
    """Every CR operand whose reference text calls it a scalar links to the section."""
    linked = []
    for slot in isa().layout.layout_order:
        for instruction in isa().instructions(slot):
            for text in instruction.doc.operands:
                named = _OPERAND_DOC.match(text)
                if not named or named["name"] not in {op.name for op in instruction.operands}:
                    continue
                operand = instruction.operand(named["name"])
                if operand.type.name in ("CrIdx", "LcrIdx") and "scalar" in text:
                    assert CR_SCALAR_LINK in text, (instruction.mnemonic, named["name"])
                    linked.append(instruction.mnemonic)
    assert sorted(linked) == ["MULT.EE", "MULT.RC.VE", "MULT.VE"]
