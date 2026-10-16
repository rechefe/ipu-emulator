"""Every generator renders the ISA it is given, not only ``isa()`` (E03-11).

E03-24 renders a modified ISA in-process; these tests add one instruction and
check each generated artifact names it, and that the default output does not.
"""

from __future__ import annotations

import dataclasses

import pytest

from ipu_as import gen_docs, gen_vscode
from ipu_gen import sv_package
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
    gen_docs.generate_instruction_docs(tmp_path / "view.md", view)
    gen_docs.generate_instruction_docs(tmp_path / "default.md")
    assert f"`{NEW}`" in (tmp_path / "view.md").read_text()
    assert NEW not in (tmp_path / "default.md").read_text()


def test_editor_data(view):
    assert NEW in gen_vscode.render_hover_data(None, view)
    assert NEW in gen_vscode.render_tmlanguage(view)
    assert NEW not in gen_vscode.render_hover_data()


def test_sv_package(view):
    assert NEW in sv_package.render_sv_package(view=view)
    assert NEW not in sv_package.render_sv_package()


def test_sv_package_takes_a_context_or_a_view():
    with pytest.raises(TypeError):
        sv_package.render_sv_package(sv_package.build_codegen_context(), view=isa())
