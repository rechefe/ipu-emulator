"""One ISA entry is enough for the generators (E03-24, Part A).

The SV package, the instruction reference and the editor data rendered from
the ISA with ``TESTNEG`` added name it, and every ``*_WIDTH`` localparam stays
as the E02-03 golden pins it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from one_edit import MNEMONIC, view_with_dummy

from ipu_gen import docs_pages, sv_package, vscode_data
from ipu_isa.slots import Slot
from ipu_isa.view import isa

GOLDEN = Path.cwd() / "src/tools/ipu-as/test/baselines/ipu_instr_pkg.sv"
_LOCALPARAM = re.compile(r"localparam int unsigned (\w+) = (\d+);")


@pytest.fixture(scope="module")
def view():
    return view_with_dummy()


def test_the_sv_package_names_it_and_keeps_every_width(view):
    text = sv_package.render_sv_package(view=view)
    opcode = len(isa().instructions(Slot.LR))  # the next free one, whatever LR has today
    bits = view.layout.unions[Slot.LR].opcode_bits
    assert f"LR_INST_OPCODE_TESTNEG = {bits}'d{opcode}" in text
    assert re.search(r"^\s*} testneg;$", text, re.MULTILINE)
    assert dict(_LOCALPARAM.findall(text)) == dict(_LOCALPARAM.findall(GOLDEN.read_text()))


def test_the_instruction_reference_documents_it(view, tmp_path):
    page = tmp_path / "instructions.md"
    docs_pages.generate_instruction_docs(page, view)
    assert f"### `{MNEMONIC}` — Test negate" in page.read_text()


def test_the_editor_data_lists_it(view):
    assert MNEMONIC in vscode_data.render_hover_data(None, view)
    assert MNEMONIC in vscode_data.render_tmlanguage(view)
