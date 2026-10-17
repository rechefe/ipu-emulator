"""The examples of How-to: add an instruction (docs/content/how-to/add-an-instruction.md).

The page includes the sections between the ``--8<--`` markers below, and the ISA entry and
handler of the "one edit" test's TESTNEG (src/tools/ipu-isa/test/one_edit.py) by line range.
These tests run the sections, and check that each line range still holds exactly that entry
and that handler, so the page shows only code that runs.
"""

from __future__ import annotations

import re
from pathlib import Path

import one_edit
import pytest

# --8<-- [start:test-import]
from emu_support import run

# --8<-- [end:test-import]
# --8<-- [start:pseudo-import]
from ipu_as.lark_tree import assemble

# --8<-- [end:pseudo-import]
from ipu_emu.slots import HANDLER_MODULES
from ipu_isa.slots import Slot

# The workspace root: the runfiles tree under Bazel, the checkout under plain pytest.
ROOT = Path(__file__).parents[2]
PAGE = ROOT / "docs" / "content" / "how-to" / "add-an-instruction.md"


# --8<-- [start:test]
def test_add_sums_two_local_registers():
    """ADD writes src_a + src_b to dest, wrapping at 32 bits."""
    state = run("SET LR0 CR2;; SET LR1 CR3;; ADD LR2 LR0 LR1;; HALT;;", cr={2: 0xFFFFFFFF, 3: 2})
    assert state.regfile.get_lr(2) == 1


# --8<-- [end:test]


# --8<-- [start:pseudo]
def test_bgt_is_blt_with_its_operands_swapped():
    assert assemble("BGT.S LR0 LR1 +1;;") == assemble("BLT.S LR1 LR0 +1;;")


# --8<-- [end:pseudo]


#: `--8<-- "file:first:last"` in the page.
_RANGE = re.compile(r'^--8<-- "(?P<file>[^":]+):(?P<first>\d+):(?P<last>\d+)"$', re.M)


def _ranges() -> dict[str, str]:
    """The text each line-range include of the page shows, by first included line."""
    found = {}
    for m in _RANGE.finditer(PAGE.read_text(encoding="utf-8")):
        assert m["file"] == "src/tools/ipu-isa/test/one_edit.py", m[0]
        lines = (ROOT / m["file"]).read_text(encoding="utf-8").split("\n")
        found[m[0]] = "\n".join(lines[int(m["first"]) - 1 : int(m["last"])]) + "\n"
    return found


@pytest.mark.parametrize(
    ("what", "source"),
    [
        ("the ISA entry", one_edit.ENTRY_SOURCE),
        ("the handler", one_edit.HANDLER_SOURCE.rstrip("\n") + "\n"),
    ],
)
def test_the_page_includes_the_one_edit_tests_own_code(what, source):
    assert source in _ranges().values(), (
        f"{what}: no include in {PAGE.name} shows one_edit.py's text; update its line range"
    )


def test_the_handler_goes_where_the_page_says():
    assert one_edit.SLOT == "lr"
    assert HANDLER_MODULES[Slot.LR].__name__ == "ipu_emu.slots.lr"
