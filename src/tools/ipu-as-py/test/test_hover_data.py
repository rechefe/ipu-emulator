"""The reference the editor shows on hover must be code that actually assembles.

`InstructionDoc.syntax` and `.example` are written with comma-separated operands
throughout, following the prose convention in CLAUDE.md -- but `asm_grammar.lark`
has no comma terminal, so that form is a lex error. The generator normalizes
those two fields; these tests are what stop that normalization from rotting, and
what stop a new instruction from shipping an example that does not assemble.
"""

import json
import re

import pytest

from ipu_as.diagnostics import check
from ipu_as.gen_vscode import build_hover_data, render_hover_data

HOVER = build_hover_data()

#: `syntax` is deliberately placeholder text (`MULT.RC.VE rc_idx src ...`), so
#: it names operands rather than supplying them and cannot assemble. Only
#: `example` is real code.
EXAMPLES = sorted(
    {
        form["doc"]["example"]
        for forms in HOVER["instructions"].values()
        for form in forms
        if form.get("doc") and form["doc"].get("example")
    }
)


def test_there_are_examples_to_check():
    # Guards the parametrization below: an empty list would pass vacuously.
    assert len(EXAMPLES) > 20


_UNDEFINED_LABEL_RE = re.compile(r"Label (\w+) not defined")


def assemble_snippet(example: str):
    """Assemble one example, supplying any label it branches to.

    A branch example naturally names a target — `BEQ LR0 LR1 end;;` — that only
    exists in the surrounding program. That is not a defect in the example, so
    define the label and retry rather than exempting branches from the check.
    """
    text = example
    for _ in range(4):
        found = check(text)
        if not found:
            return []
        names = {n for d in found for n in _UNDEFINED_LABEL_RE.findall(d.message)}
        if not names:
            return found
        text = text.rstrip("\n") + "\n" + "".join(f"{n}: BKPT;;\n" for n in sorted(names))
    return check(text)


@pytest.mark.parametrize("example", EXAMPLES)
def test_every_shipped_example_assembles(example):
    found = assemble_snippet(example)
    assert not found, f"{example!r} does not assemble: {[d.message for d in found]}"


@pytest.mark.parametrize(
    "field", ["syntax", "example"]
)
def test_code_fields_use_the_separator_the_grammar_accepts(field):
    # Operands are whitespace-separated; a comma would not lex.
    offenders = [
        form["doc"][field]
        for forms in HOVER["instructions"].values()
        for form in forms
        if form.get("doc") and "," in (form["doc"].get(field) or "")
    ]
    assert not offenders, f"{field} still contains commas: {offenders[:3]}"


def test_every_mnemonic_has_hover_text():
    missing = [m for m, forms in HOVER["instructions"].items() if not forms[0].get("doc")]
    assert not missing, f"no documentation for: {missing}"


def test_rendering_is_deterministic():
    assert render_hover_data() == render_hover_data()


def test_registers_are_present():
    # Hover covers registers too; lrd pairs live in ipu_as.reg, not the master
    # register table, and have been missed before.
    registers = set(HOVER["registers"])
    assert {"lr0", "cr15", "r0", "lrd0", "lrd14"} <= registers
