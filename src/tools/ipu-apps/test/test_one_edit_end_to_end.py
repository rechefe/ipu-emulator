"""One ISA entry and one handler are enough, end to end (E03-24, Part B).

Copies the ``ipu_isa``, ``ipu_as`` and ``ipu_emu`` sources to a temporary
directory, makes exactly two edits (the ``TESTNEG`` entry into
``instruction_spec.py``, its handler into ``Ipu``), then assembles and runs
``SET LR0 CR2 ;; TESTNEG LR1 LR0 ;; HALT ;;`` in a fresh interpreter that
imports the copies. ``KNOWN_MANUAL_EDITS`` lists any further edit adding an
instruction still needs; it is empty, and a listed edit that is no longer
needed fails the test, so the list can only shrink.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from one_edit import ENTRY_SOURCE, HANDLER_SOURCE

import ipu_as
import ipu_emu
import ipu_isa

#: (file under the copied packages, old text, new text, reason and the issue
#: that removes it). Empty: E03-11 made the spec the only ISA input.
KNOWN_MANUAL_EDITS: list[tuple[str, str, str, str]] = []

SPEC = "ipu_isa/instruction_spec.py"
HANDLERS = "ipu_emu/ipu.py"

#: Where each edit goes: the end of the spec's "lr" dict, and the start of
#: the LR handlers.
_LR_OPEN = '    "lr": {\n'
_SLOT_CLOSE = "\n    },\n"
_HANDLER_ANCHOR = "    def execute_lr_add(self"

PROGRAM = "SET LR0 CR2 ;; TESTNEG LR1 LR0 ;; HALT ;;"

RUN = f"""
import ipu_as
import ipu_emu
import ipu_isa
from ipu_as.lark_tree import assemble
from ipu_emu.emulator import load_program, run_until_complete
from ipu_emu.execute import decode_instruction_word
from ipu_emu.ipu_state import IpuState

for package in (ipu_isa, ipu_as, ipu_emu):
    assert package.__file__.startswith({{root!r}}), package.__file__
state = IpuState()
state.regfile.set_cr(2, 5)
load_program(state, [decode_instruction_word(word) for word in assemble({PROGRAM!r})])
run_until_complete(state)
print(state.regfile.get_lr(1))
"""


def _copy_sources(root: Path) -> None:
    for package in (ipu_isa, ipu_as, ipu_emu):
        source = Path(package.__file__).parent
        shutil.copytree(source, root / source.name, ignore=shutil.ignore_patterns("__pycache__"))


def _insert_entry(root: Path) -> None:
    path = root / SPEC
    text = path.read_text(encoding="utf-8")
    start = text.index(_LR_OPEN)
    end = text.index(_SLOT_CLOSE, start)
    path.write_text(text[: end + 1] + ENTRY_SOURCE + text[end + 1 :], encoding="utf-8")


def _insert_handler(root: Path) -> None:
    path = root / HANDLERS
    text = path.read_text(encoding="utf-8")
    at = text.index(_HANDLER_ANCHOR)
    path.write_text(text[:at] + HANDLER_SOURCE + text[at:], encoding="utf-8")


def _apply_manual_edits(root: Path, skip: int | None = None) -> None:
    for index, (file, old, new, _) in enumerate(KNOWN_MANUAL_EDITS):
        if index == skip:
            continue
        path = root / file
        text = path.read_text(encoding="utf-8")
        assert text.count(old) == 1, f"{file}: the listed edit no longer applies"
        path.write_text(text.replace(old, new), encoding="utf-8")


def _run(root: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(root), *sys.path]))
    return subprocess.run(
        [sys.executable, "-c", RUN.format(root=str(root))],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


@pytest.fixture
def root(tmp_path: Path) -> Path:
    _copy_sources(tmp_path)
    return tmp_path


def test_two_edits_add_a_working_instruction(root: Path):
    _insert_entry(root)
    _insert_handler(root)
    _apply_manual_edits(root)
    result = _run(root)
    assert result.returncode == 0, result.stderr
    assert int(result.stdout) == -5 & 0xFFFFFFFF


def test_without_the_handler_the_emulator_names_the_missing_handler(root: Path):
    _insert_entry(root)
    _apply_manual_edits(root)
    result = _run(root)
    assert result.returncode != 0
    assert "names handler Ipu.execute_testneg, which does not exist" in result.stderr


@pytest.mark.parametrize("index", range(len(KNOWN_MANUAL_EDITS)))
def test_every_listed_manual_edit_is_still_needed(root: Path, index: int):
    _insert_entry(root)
    _insert_handler(root)
    _apply_manual_edits(root, skip=index)
    assert _run(root).returncode != 0, f"{KNOWN_MANUAL_EDITS[index][0]} is no longer needed"
