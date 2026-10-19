"""Every generator renders the ISA it is given, not only ``isa()`` (E03-11).

E03-24 renders a modified ISA in-process; these tests add one instruction and
check each generated artifact names it, and that the default output does not.
"""

from __future__ import annotations

import copy
import dataclasses

import pytest

from ipu_gen import docs_pages, sv_package, vscode_data
from ipu_isa.operand_types import OPERAND_TYPES
from ipu_isa.slots import SLOTS
from ipu_isa.view import Isa, authoring_copies, isa, load_isa

NEW = "TESTNEG"


@pytest.fixture(scope="module")
def view() -> Isa:
    spec, pseudo = authoring_copies()
    entry = spec["lr"]["ADD"]
    entry["doc"] = dataclasses.replace(entry["doc"], example=f"{NEW} LR0 LR1 LR2;;")
    spec["lr"][NEW] = entry
    return load_isa(spec, pseudo, SLOTS, OPERAND_TYPES)


def test_the_view_has_the_new_instruction(view):
    assert NEW in {i.mnemonic for i in view.instructions(view.find_slot(NEW))}


def test_instruction_reference(view, tmp_path):
    docs_pages.generate_instruction_docs(tmp_path / "view.md", view)
    docs_pages.generate_instruction_docs(tmp_path / "default.md")
    assert f"`{NEW}`" in (tmp_path / "view.md").read_text()
    assert NEW not in (tmp_path / "default.md").read_text()


def test_editor_data(view):
    assert NEW in vscode_data.render_hover_data(None, view)
    assert NEW in vscode_data.render_tmlanguage(view)
    assert NEW not in vscode_data.render_hover_data()


def test_sv_package(view):
    assert NEW in sv_package.render_sv_package(view=view)
    assert NEW not in sv_package.render_sv_package()


def test_sv_package_takes_a_context_or_a_view():
    with pytest.raises(TypeError):
        sv_package.render_sv_package(sv_package.build_codegen_context(), view=isa())


def test_an_enum_only_an_emulator_slot_uses_is_left_out():
    # ElementsInRow has one user, ACC.STRIDE (ACC, hardware). Move its only use
    # to an emulator-only BREAK instruction: its typedef must leave the package.
    spec, pseudo = authoring_copies()
    del spec["acc"]["ACC.STRIDE"]
    entry = copy.deepcopy(spec["break"]["BREAK"])
    entry["operands"] = [{"name": "row", "type": "ElementsInRow"}]
    entry["doc"] = dataclasses.replace(
        entry["doc"], operands=["row: elements per row"], example="BREAK.ROW 32;;"
    )
    spec["break"]["BREAK.ROW"] = entry
    view = load_isa(spec, pseudo, SLOTS, OPERAND_TYPES)
    assert view.instruction(view.find_slot("BREAK.ROW"), "BREAK.ROW").operands[0].type.name == (
        "ElementsInRow"
    )
    assert "elements_in_row_field_t" in sv_package.render_sv_package()
    assert "elements_in_row" not in sv_package.render_sv_package(view=view)


def test_a_reserved_word_mnemonic_gets_a_suffix():
    # FINAL is a SystemVerilog keyword, so its union member is final_inst.
    spec, pseudo = authoring_copies()
    entry = copy.deepcopy(spec["lr"]["ADD"])
    entry["doc"] = dataclasses.replace(entry["doc"], example="FINAL LR0 LR1 LR2;;")
    spec["lr"]["FINAL"] = entry
    text = sv_package.render_sv_package(view=load_isa(spec, pseudo, SLOTS, OPERAND_TYPES))
    assert "} final_inst;" in text
    assert "} final;" not in text
