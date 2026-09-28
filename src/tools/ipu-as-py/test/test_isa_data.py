"""The editor's completion data (`isa.json`: per operand type, the values or
range the editor offers, from each token class's `completion_domain()`) held
against the constructors the assembler validates with, so the editor can never
suggest something that fails to assemble, nor miss a boundary."""

import contextlib

import lark
import pytest

from ipu_common.acc_stride_enums import get_horizontal_stride_bits, get_vertical_stride_bits
from ipu_common.instruction_spec import VALID_OPERAND_TYPES
from ipu_as.diagnostics import check
from ipu_as.gen_vscode import build_hover_data
from ipu_as.inst import OPERAND_TYPE_MAP
from ipu_as.ipu_token import AnnotatedToken
from ipu_as.lark_tree import get_parser

DATA = build_hover_data()
TYPES = DATA["operandTypes"]
RELATIVE_MAX = TYPES["Label"]["relative_max"]


def _build(type_name: str, text: str, instr_id: int = 0):
    return OPERAND_TYPE_MAP[type_name](AnnotatedToken(lark.Token("TOKEN", text), instr_id))


def test_every_operand_type_is_described():
    assert set(TYPES) == set(VALID_OPERAND_TYPES) == set(OPERAND_TYPE_MAP)
    for name, domain in TYPES.items():
        assert domain["description"], name


@pytest.mark.parametrize(
    "type_name, value", [(name, value) for name, domain in TYPES.items() for value in domain.get("values", [])]
)
def test_every_offered_value_is_accepted(type_name, value):
    _build(type_name, value)
    if DATA["lexical"]["caseInsensitive"]["registers"]:
        _build(type_name, value.upper())


@pytest.mark.parametrize(
    "type_name, edge, beyond",
    [(name, domain[end], domain[end] + step) for name, domain in TYPES.items() if "min" in domain
     for end, step in (("min", -1), ("max", 1))] + [("Label", f"+{RELATIVE_MAX}", f"+{RELATIVE_MAX + 1}")],
)
def test_range_is_exact(type_name, edge, beyond):
    # Each end of a numeric range (and a label's relative offset) is accepted;
    # one step past it is not.
    _build(type_name, str(edge))
    with pytest.raises(ValueError):
        _build(type_name, str(beyond))


@pytest.mark.parametrize(
    "type_name, decode",
    [("HorizontalStride", get_horizontal_stride_bits), ("VerticalStride", get_vertical_stride_bits)],
)
def test_withheld_enum_values_are_the_undecodable_ones(type_name, decode):
    # The assembler accepts padding names like `reserved3`; completion leaves
    # them out because the emulator cannot execute them.
    offered = TYPES[type_name]["values"]
    for index, name in enumerate(OPERAND_TYPE_MAP[type_name].enum_array()):
        with contextlib.nullcontext() if name in offered else pytest.raises(KeyError):
            decode(index)


def _operand_text(type_name: str) -> str:
    domain = TYPES[type_name]
    if domain["kind"] == "label":
        return "+0"
    return domain["values"][-1] if domain.get("values") else str(domain["max"])


@pytest.mark.parametrize(
    "mnemonic, form",
    [(mnemonic, form) for mnemonic, forms in DATA["instructions"].items() for form in forms if form["operands"]],
    ids=lambda v: v if isinstance(v, str) else v["slot"],
)
def test_every_instruction_assembles_from_completion_values(mnemonic, form):
    # End to end: an instruction written purely from what completion offers
    # must pass the same check the editor's squiggles run.
    operands = " ".join(_operand_text(op["type"]) for op in form["operands"])
    source = f"{mnemonic} {operands};;\nBKPT;;\n"
    assert check(source) == [], source


def test_lexical_facts_match_the_grammar_and_the_assembler():
    lexical = DATA["lexical"]
    terminals = {t.name: t.pattern.to_regexp() for t in get_parser().terminals}
    assert lexical["token"] == terminals["TOKEN"]
    assert set(lexical["comments"]) == {terminals["COMMENT_LINE"], terminals["COMMENT_HASH"]}
    assert lexical["caseInsensitive"] == {"mnemonics": True, "registers": True, "labels": False}
    assert check("Loop:\n    B Loop;;\nBKPT;;\n") == []
    assert check("Loop:\n    B loop;;\nBKPT;;\n") != []


def test_every_register_is_described_by_its_definition():
    # The hover's words come from ipu_common.registers (and LrdRegField for
    # pairs), and the fixed values from whoever passes them (the emulator).
    data = build_hover_data({"fixed": {"cr0": 0, "cr1": 1}, "pairs": {"lrd4": (4, 5)}})
    docs = data["registerDocs"]
    assert set(docs) == set(data["registers"])
    assert all(doc["about"] for doc in docs.values())
    assert docs["cr0"]["fixed"] == 0 and docs["cr1"]["fixed"] == 1
    assert "fixed" not in docs["cr2"]
    assert docs["lrd4"]["pair"] == ["lr5", "lr4"]  # hi:lo, as the emulator pairs them
