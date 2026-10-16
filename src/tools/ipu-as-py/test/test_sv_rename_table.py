"""The SV rename table of the mnemonic standard (E03-19) matches the package.

The table is published for the downstream RTL repository in
``docs/content/specs/mnemonic-rename.md``. Each new enum member must carry the
old member's value, each new union member must exist, and no old name may
remain in the golden ``ipu_instr_pkg.sv``. Mapping every new name back to
its old one gives the package as it was before the rename, byte for byte
(but for the content-hash header): the table names every identifier that
changed.

E03-22 changes the package (hardware slots only); it updates
``PRE_RENAME_SHA256`` or retires that test.
"""

from __future__ import annotations

import hashlib
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

#: sha256 of ipu_instr_pkg.sv before E03-19, without its line 2 (the content hash).
PRE_RENAME_SHA256 = "dce1f7932b6f26ae01c19ecde8ad062196411acb6b139f5814d75e7016771d5c"


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


def test_the_table_names_every_changed_identifier():
    lines = GOLDEN.read_text(encoding="utf-8").splitlines(keepends=True)
    assert lines[1].startswith("// Content: sha256:")
    text = "".join(lines[:1] + lines[2:])
    for _, _, old_enum, new_enum, old_member, new_member, _ in ROWS:
        text = re.sub(rf"\b{new_enum}\b", old_enum, text)
        text = re.sub(rf"}} {new_member};", f"}} {old_member};", text)
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == PRE_RENAME_SHA256
