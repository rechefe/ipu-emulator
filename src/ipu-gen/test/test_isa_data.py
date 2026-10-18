"""The editor's completion data (`isa.json`: per operand type, the values or
range the editor offers, from each token class's `completion_domain()`) held
against the constructors the assembler validates with, so the editor can never
suggest something that fails to assemble, nor miss a boundary."""

import contextlib
import itertools
import random
import re
from pathlib import Path

import lark
import pytest

from ipu_as.diagnostics import check
from ipu_as.inst import OPERAND_TYPE_MAP
from ipu_as.ipu_token import AnnotatedToken
from ipu_as.lark_tree import get_parser
from ipu_gen.vscode_data import build_hover_data
from ipu_isa.acc_stride_enums import get_horizontal_stride_bits, get_vertical_stride_bits
from ipu_isa.instruction_spec import VALID_OPERAND_TYPES

DATA = build_hover_data()
TYPES = DATA["operandTypes"]
RELATIVE_MAX = TYPES["Label"]["relative_max"]
# The workspace root: the runfiles tree under Bazel, the checkout under plain pytest.
CORPUS = Path(__file__).parents[3] / "vscode-ipu-asm" / "test" / "corpus"


def _build(type_name: str, text: str, instr_id: int = 0):
    return OPERAND_TYPE_MAP[type_name](AnnotatedToken(lark.Token("TOKEN", text), instr_id))


def test_every_operand_type_is_described():
    assert set(TYPES) == set(VALID_OPERAND_TYPES) == set(OPERAND_TYPE_MAP)
    for name, domain in TYPES.items():
        assert domain["description"], name


@pytest.mark.parametrize(
    "type_name, value",
    [(name, value) for name, domain in TYPES.items() for value in domain.get("values", [])],
)
def test_every_offered_value_is_accepted(type_name, value):
    _build(type_name, value)
    if DATA["lexical"]["caseInsensitive"]["registers"]:
        _build(type_name, value.upper())


@pytest.mark.parametrize(
    "type_name, edge, beyond",
    [
        (name, domain[end], domain[end] + step)
        for name, domain in TYPES.items()
        if "min" in domain
        for end, step in (("min", -1), ("max", 1))
    ]
    + [("Label", f"+{RELATIVE_MAX}", f"+{RELATIVE_MAX + 1}")],
)
def test_range_is_exact(type_name, edge, beyond):
    # Each end of a numeric range (and a label's relative offset) is accepted;
    # one step past it is not.
    _build(type_name, str(edge))
    with pytest.raises(ValueError):
        _build(type_name, str(beyond))


@pytest.mark.parametrize(
    "type_name, decode",
    [
        ("HorizontalStride", get_horizontal_stride_bits),
        ("VerticalStride", get_vertical_stride_bits),
    ],
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
    [
        (mnemonic, form)
        for mnemonic, forms in DATA["instructions"].items()
        for form in forms
        if form["operands"]
    ],
    ids=lambda v: v if isinstance(v, str) else v["slot"],
)
def test_every_instruction_assembles_from_completion_values(mnemonic, form):
    # End to end: an instruction written purely from what completion offers
    # must pass the same check the editor's squiggles run.
    operands = " ".join(_operand_text(op["type"]) for op in form["operands"])
    source = f"{mnemonic} {operands};;\nHALT;;\n"
    assert check(source) == [], source


def test_lexical_facts_match_the_grammar_and_the_assembler():
    lexical = DATA["lexical"]
    terminals = {t.name: t.pattern.to_regexp() for t in get_parser().terminals}
    assert lexical["token"] == terminals["TOKEN"]
    assert set(lexical["comments"]) == {terminals["COMMENT_LINE"], terminals["COMMENT_HASH"]}
    assert lexical["caseInsensitive"] == {"mnemonics": True, "registers": True, "labels": False}
    assert check("Loop:\n    B Loop;;\nHALT;;\n") == []
    assert check("Loop:\n    B loop;;\nHALT;;\n") != []


def test_every_register_is_described_by_its_definition():
    # The hover's words come from ipu_isa.registers (and LrdRegField for
    # pairs), and the fixed values and LRD pairs from the facts passed in
    # (vscode_data.REGISTER_FACTS in the generated file).
    data = build_hover_data({"fixed": {"cr0": 0, "cr1": 1}, "pairs": {"lrd4": (4, 5)}})
    docs = data["registerDocs"]
    assert set(docs) == set(data["registers"])
    assert all(doc["about"] for doc in docs.values())
    assert docs["cr0"]["fixed"] == 0 and docs["cr1"]["fixed"] == 1
    assert "fixed" not in docs["cr2"]
    assert docs["lrd4"]["pair"] == ["lr5", "lr4"]  # hi:lo, as ipu_isa.registers.lrd_pair pairs them


def _literals():
    """Every string of up to 4 characters over the characters a number literal
    can contain (and a few it cannot), plus longer seeded random ones."""
    alphabet = "+-0179aCfgxXoObB_."
    for length in range(1, 5):
        yield from map("".join, itertools.product(alphabet, repeat=length))
    rng = random.Random(0)
    for _ in range(20000):
        yield "".join(rng.choice(alphabet) for _ in range(rng.randint(5, 10)))


def _int_accepts(text: str) -> bool:
    try:
        int(text, 0)
    except ValueError:
        return False
    return True


def _corpus_tokens():
    """Every token of the extension's assembly corpus."""
    for path in sorted(CORPUS.rglob("*.asm")):
        yield from re.findall(r"[^\s;:]+", path.read_text(encoding="utf-8"))


def test_schema_version():
    assert DATA["schemaVersion"] == 1


@pytest.mark.parametrize("source", ["generated", "corpus"])
def test_number_pattern_agrees_with_int_base_0(source):
    # The editor's numberPattern (a hand-written regex, in syntax Python's re and
    # JavaScript's RegExp read alike) accepts exactly what NumberToken's
    # int(v, 0) does, among the strings the grammar's TOKEN can carry to it.
    number = re.compile(DATA["numberPattern"])
    token = re.compile(DATA["lexical"]["token"])
    texts = list(_literals() if source == "generated" else _corpus_tokens())
    assert texts
    disagree = [
        text
        for text in texts
        if token.fullmatch(text) and bool(number.fullmatch(text)) != _int_accepts(text)
    ]
    assert disagree == []


def _word(mnemonic: str) -> str:
    form = DATA["instructions"][mnemonic][0]
    return " ".join([mnemonic, *(_operand_text(op["type"]) for op in form["operands"])])


def test_slot_fill_is_the_assemblers():
    # A wildcard (NOP) is in every slot; every other mnemonic has one kind, the
    # slot it is specified in, or for a pseudo-instruction its expansion's.
    fill = DATA["slotFill"]
    assert fill["order"] == DATA["slotOrder"] and fill["rule"] == "own-kind"
    assert fill["wildcards"] == ["nop"]
    for mnemonic, forms in DATA["instructions"].items():
        if mnemonic.lower() in fill["wildcards"]:
            assert {f["slot"] for f in forms} == set(fill["order"])
        else:
            assert fill["kinds"][mnemonic] == forms[0]["slot"], mnemonic
    # Two instructions of one kind in a word that holds one are rejected, as
    # the editor's fits() rejects them; a wildcard then takes another slot.
    one_slot = next(k for k in fill["order"] if DATA["slotCount"][k] == 1 and k != "break")
    same_kind = [m for m, k in fill["kinds"].items() if k == one_slot]
    assert len(same_kind) >= 2, f"the {one_slot} slot has fewer than two mnemonics"
    first, second = same_kind[:2]
    assert check(f"{_word(first)}; {_word(second)};;\nHALT;;\n") != []
    assert check(f"{_word(first)}; NOP;;\nHALT;;\n") == []
