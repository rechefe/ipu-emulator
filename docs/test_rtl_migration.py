"""specs/rtl-migration.md against the SystemVerilog golden of the current package.

The page tells the downstream RTL repository what changed in ipu_instr_pkg.sv
since master. These checks keep its claims about the current package true: each new
name in its rename table is in the package with its value and each old one is
gone, the names it says were removed are gone, and the word width it states
is the package's.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
PAGE = (ROOT / "docs" / "content" / "specs" / "rtl-migration.md").read_text(encoding="utf-8")
GOLDEN = (
    ROOT / "src" / "tools" / "ipu-as-py" / "test" / "baselines" / "ipu_instr_pkg.sv"
).read_text(encoding="utf-8")

#: | `old enum` | `new enum` | `old union` | `new union` | `value` |
_ROW = re.compile(r"^\| `(\w+)` \| `(\w+)` \| `(\w+)` \| `(\w+)` \| `(\d+'d\d+)` \|$", re.M)


def _identifiers(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", text))


def _removed() -> list[str]:
    section = PAGE[PAGE.index("### 2.") : PAGE.index("### 3.")]
    intro = section[: section.index("So `IPU_COMPOUND_INST_WIDTH`")]
    # The struct that loses the two members is named there too, and stays.
    return [name for name in re.findall(r"`(\w+)`", intro) if name != "ipu_compound_inst_t"]


@pytest.mark.parametrize("row", _ROW.findall(PAGE), ids=lambda row: row[1])
def test_each_renamed_member_has_its_new_name_and_value(row):
    old_enum, new_enum, old_union, new_union, value = row
    assert f"{new_enum} = {value}" in GOLDEN
    assert re.search(rf"\}} {new_union};", GOLDEN)
    assert not {old_enum, old_union} & _identifiers(GOLDEN)


def test_the_table_lists_eight_members():
    assert len(_ROW.findall(PAGE)) == 8


def test_the_removed_names_are_gone():
    removed = _removed()
    assert "acc_store_slot_t" in removed and "BREAK_SLOT_T_WIDTH" in removed
    assert not set(removed) & _identifiers(GOLDEN)


def test_the_word_width_is_the_packages():
    assert "localparam int unsigned IPU_COMPOUND_INST_WIDTH = 155;" in GOLDEN
    assert "`IPU_COMPOUND_INST_WIDTH` goes from 186 to 155" in PAGE
