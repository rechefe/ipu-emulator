"""The SV rename table of the mnemonic standard (E03-19) matches the package.

The table is published for the downstream RTL repository in
``docs/content/specs/mnemonic-rename.md``. Each new enum member must carry the
old member's value, each new union member must exist, and no old name may
remain in the golden ``ipu_instr_pkg.sv``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path.cwd()
TABLE = ROOT / "docs/content/specs/mnemonic-rename.md"
GOLDEN = ROOT / "src/tools/ipu-as-py/test/golden/ipu_instr_pkg.sv"


def _rows() -> list[list[str]]:
    rows = []
    for line in TABLE.read_text(encoding="utf-8").splitlines():
        cells = [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]
        if len(cells) == 7 and cells[0] not in ("Old mnemonic", "---"):
            rows.append(cells)
    return rows


ROWS = _rows()


def test_the_table_has_every_renamed_instruction():
    assert len(ROWS) == 9


@pytest.mark.parametrize("row", ROWS, ids=[row[1] for row in ROWS])
def test_each_row_matches_the_package(row):
    _, _, old_enum, new_enum, old_member, new_member, value = row
    package = GOLDEN.read_text(encoding="utf-8")
    assert re.search(rf"^\s*{new_enum} = {re.escape(value)},?$", package, re.MULTILINE)
    assert re.search(rf"^\s*}} {new_member};$", package, re.MULTILINE)
    assert not re.search(rf"\b{old_enum}\b", package)
    assert not re.search(rf"}} {old_member};", package)
