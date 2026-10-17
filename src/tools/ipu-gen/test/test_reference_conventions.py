"""The generated ISA reference follows the CLAUDE.md conventions (E09-14).

Register names are upper case (``LR0``, not ``lr0``); syntax lines separate
operands with spaces, not commas; pseudocode gives sizes in elements, not
bytes. SVG diagrams are skipped: their labels are field names.
"""

from __future__ import annotations

import re

import pytest

from ipu_gen import docs_pages

_LOWER_REGISTER = re.compile(r"(?<![A-Za-z0-9_.\-])(?:r|lr|cr|lrd)\d+\b")
_BYTES = re.compile(r"\bbytes?\b")  # the unit, not an identifier such as byte_val


def violations(text: str) -> list[str]:
    """Every convention break in one generated page, as ``line N: reason``."""
    found = []
    lines = text.splitlines()
    for number, line in enumerate(lines, start=1):
        if line.lstrip().startswith("<"):
            continue  # SVG markup
        if _LOWER_REGISTER.search(line):
            found.append(f"line {number}: lower-case register {_LOWER_REGISTER.search(line)[0]}")
        if line.startswith("**Syntax:**") and "," in line:
            found.append(f"line {number}: comma in a syntax line")
        if number > 1 and lines[number - 2].startswith("**Pseudo code:**") and _BYTES.search(line):
            found.append(f"line {number}: pseudocode size in bytes")
    return found


@pytest.fixture(scope="module")
def pages(tmp_path_factory) -> dict[str, str]:
    out = tmp_path_factory.mktemp("reference")
    docs_pages.generate_all_docs(out)
    docs_pages.generate_instruction_format_md(out / "instruction-format.md")
    return {path.name: path.read_text() for path in sorted(out.glob("*.md"))}


def test_every_page_is_checked(pages):
    assert set(pages) == {*docs_pages.REFERENCE_PAGES, "instruction-format.md"}


@pytest.mark.parametrize("name", [*docs_pages.REFERENCE_PAGES, "instruction-format.md"])
def test_page_follows_the_conventions(pages, name):
    assert violations(pages[name]) == []


@pytest.mark.parametrize(
    ("planted", "reason"),
    [
        ("Load into lr0.", "lower-case register lr0"),
        ("**Syntax:** `BEQ reg1, reg2, label`", "comma in a syntax line"),
        ("**Pseudo code:**\n`R_ACC = x (512 bytes)`", "pseudocode size in bytes"),
    ],
)
def test_a_planted_violation_is_found(planted, reason):
    assert any(reason in v for v in violations("# Page\n\n" + planted + "\n"))


def test_identifiers_and_upper_case_registers_pass():
    assert violations("Write `LR0`, `CR15` and `R1`; the field `lr_reg_field_t`; FP32.\n") == []
    assert violations("**Pseudo code:**\n`byte_val = imm & 0xFF`\n") == []
