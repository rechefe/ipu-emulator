"""The generated SV package matches its golden file, and every ``*_WIDTH``
localparam matches the table below (E02-03).

A downstream RTL repo vendors ``ipu_instr_pkg.sv`` verbatim, so any change to
the rendered text is a change to what they get. The golden is rendered with the
commit unstamped (``source_commit="unknown"``); regenerate it with
``bazel run //src/tools/ipu-as-py:update_sv_golden``. Every ``*_WIDTH`` is wire
format: changing one means editing ``EXPECTED_LOCALPARAMS`` by hand and calling
the change out in the PR as breaking (CLAUDE.md, decision D6).
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path

import update_sv_golden

GOLDEN = Path.cwd() / update_sv_golden.GOLDEN_PATH

#: Every localparam of the package, by name. Edit only on purpose.
EXPECTED_LOCALPARAMS = {
    "LOAD_SLOT_U_WIDTH": 12,
    "LOAD_SLOT_T_WIDTH": 14,
    "STORE_SLOT_U_WIDTH": 8,
    "STORE_SLOT_T_WIDTH": 9,
    "ACC_STORE_SLOT_U_WIDTH": 8,
    "ACC_STORE_SLOT_T_WIDTH": 9,
    "LR_SLOT_U_WIDTH": 16,
    "LR_SLOT_T_WIDTH": 20,
    "MULT_SLOT_U_WIDTH": 20,
    "MULT_SLOT_T_WIDTH": 23,
    "ACC_SLOT_U_WIDTH": 13,
    "ACC_SLOT_T_WIDTH": 17,
    "AAQ_SLOT_U_WIDTH": 8,
    "AAQ_SLOT_T_WIDTH": 9,
    "COND_SLOT_U_WIDTH": 20,
    "COND_SLOT_T_WIDTH": 23,
    "BREAK_SLOT_U_WIDTH": 20,
    "BREAK_SLOT_T_WIDTH": 22,
    "IPU_COMPOUND_INST_WIDTH": 186,
}

_LOCALPARAM = re.compile(r"^\s*localparam\s+int\s+unsigned\s+(\w+)\s*=\s*(\d+)\s*;", re.M)


def localparams(text: str) -> dict[str, int]:
    return {name: int(value) for name, value in _LOCALPARAM.findall(text)}


def localparam_changes(expected: dict[str, int], actual: dict[str, int]) -> list[str]:
    """One line per changed, added or removed localparam, e.g.
    ``IPU_COMPOUND_INST_WIDTH: 186 -> 155``."""
    changes = []
    for name in sorted(expected.keys() | actual.keys()):
        if name not in actual:
            changes.append(f"{name}: removed (was {expected[name]})")
        elif name not in expected:
            changes.append(f"{name}: added ({actual[name]})")
        elif expected[name] != actual[name]:
            changes.append(f"{name}: {expected[name]} -> {actual[name]}")
    return changes


def test_package_text_matches_golden():
    expected = GOLDEN.read_text(encoding="utf-8")
    actual = update_sv_golden.render()
    diff = "".join(
        difflib.unified_diff(
            expected.splitlines(keepends=True),
            actual.splitlines(keepends=True),
            "ipu_instr_pkg.sv (golden)",
            "ipu_instr_pkg.sv (current code)",
        )
    )
    assert not diff, (
        "The generated SV package changed. If intended, run "
        f"bazel run //src/tools/ipu-as-py:update_sv_golden\n{diff}"
    )


def test_localparams_match_table():
    changes = localparam_changes(EXPECTED_LOCALPARAMS, localparams(update_sv_golden.render()))
    assert not changes, "SV localparams changed (wire format):\n" + "\n".join(changes)


def test_golden_localparams_match_table():
    changes = localparam_changes(
        EXPECTED_LOCALPARAMS, localparams(GOLDEN.read_text(encoding="utf-8"))
    )
    assert not changes, "Golden file and EXPECTED_LOCALPARAMS disagree:\n" + "\n".join(changes)


def test_change_message_names_each_localparam():
    actual = dict(EXPECTED_LOCALPARAMS, LR_SLOT_T_WIDTH=21, IPU_COMPOUND_INST_WIDTH=187)
    del actual["AAQ_SLOT_U_WIDTH"]
    actual["NEW_WIDTH"] = 1
    assert localparam_changes(EXPECTED_LOCALPARAMS, actual) == [
        "AAQ_SLOT_U_WIDTH: removed (was 8)",
        "IPU_COMPOUND_INST_WIDTH: 186 -> 187",
        "LR_SLOT_T_WIDTH: 20 -> 21",
        "NEW_WIDTH: added (1)",
    ]
