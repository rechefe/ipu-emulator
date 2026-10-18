"""The syntax line is derived from the mnemonic and the operand names (E03-04)."""

from __future__ import annotations

import pytest

from ipu_isa.instruction_spec import INSTRUCTION_SPEC, PSEUDO_INSTRUCTION_SPEC, syntax


def test_syntax_joins_mnemonic_and_operands_with_spaces():
    assert syntax("SET", ["reg", "src"]) == "SET reg src"
    assert syntax("NOP", []) == "NOP"


_FORMS = [
    (mnemonic, entry["operands"])
    for instructions in INSTRUCTION_SPEC.values()
    for mnemonic, entry in instructions.items()
] + [(mnemonic, entry["operands"]) for mnemonic, entry in PSEUDO_INSTRUCTION_SPEC.items()]


@pytest.mark.parametrize(("mnemonic", "operands"), _FORMS, ids=[m for m, _ in _FORMS])
def test_derived_syntax_names_exactly_the_operands(mnemonic, operands):
    first, *rest = syntax(mnemonic, [operand["name"] for operand in operands]).split(" ")
    assert first == mnemonic
    assert rest == [operand["name"] for operand in operands]
