"""ipu_as.language_service and `ipu-as check --json --json-version 2 --tokens --words --jinja` (E08-22).

Expected values come from hand-written tables for the extension's edge-case
corpus and small sources, and, over every kernel, from independent readings: the
grammar's lexer run on one source line alone, the parse tree's roles
(`dump_parser_tokens.token_roles`), the session's compound list, the ISA's
mnemonics, and a plain scan for Jinja delimiters.
"""

from __future__ import annotations

import functools
import json
import re
import time
import warnings
from pathlib import Path

import jsonschema
import pytest
from check_corpus import ROOT, sources
from click.testing import CliRunner
from referencing import Registry, Resource

from ipu_as import diagnostics, template
from ipu_as.cli import cli
from ipu_as.dump_parser_tokens import jinja_regions, token_roles
from ipu_as.language_service import FEATURES, JinjaRegion, Range, Token, analyse
from ipu_as.session import AssemblySession
from ipu_as.syntax import get_parser, parse_tree
from ipu_isa.view import isa

EDGE_CASES = ROOT / "vscode-ipu-asm" / "test" / "corpus" / "edge_cases.asm"
KERNELS = ROOT / "src" / "tools" / "ipu-apps" / "src" / "ipu_apps" / "kernels"
SCHEMAS = ROOT / "vscode-ipu-asm" / "schema"
ALL = frozenset(FEATURES)

#: The performance budget (spec: under 300 ms on CI, tracked). The assertion is
#: looser, so a slow or busy runner does not fail the build; the measured value
#: is printed and recorded as a test property.
BUDGET_MS = 300
HARD_LIMIT_MS = 1000


def _renders_alone(path: Path) -> bool:
    """False for a source whose ipu:params header has a required parameter."""
    spec = template.declared_params(path.read_text(encoding="utf-8"))
    return spec is None or all(param.default is not template.REQUIRED for param in spec.params)


def _corpus() -> list[Path]:
    """Every kernel source and the extension's assembling fixtures that render without defines."""
    paths = sorted(KERNELS.rglob("*.asm")) + sorted(
        (ROOT / "src" / "tools" / "ipu-apps" / "test").rglob("*.asm")
    )
    paths += [
        p
        for p in sorted((ROOT / "vscode-ipu-asm" / "test").rglob("*.asm"))
        if "negative" not in p.parts
    ]
    return [path for path in paths if _renders_alone(path)]


CORPUS = _corpus()


def _id(path: Path) -> str:
    return str(path.relative_to(ROOT))


@functools.cache
def _analysed(path: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # simulation-only slot warnings
        return analyse(path.read_text(encoding="utf-8"), want=ALL)


def _text(source: str, token: Token) -> str:
    return source.split("\n")[token.line][token.col : token.col + token.length]


def _tokens(source: str, *, line: int | None = None) -> list[tuple]:
    found = analyse(source, want={"tokens"}).tokens or ()
    return [
        (t.line, t.col, _text(source, t), t.role) for t in found if line is None or t.line == line
    ]


# --- analyse ---


def test_only_what_is_asked_for_is_computed():
    result = analyse("HALT;;\n", want={"tokens"})
    assert result.tokens is not None
    assert (result.diagnostics, result.words, result.jinja) == (None, None, None)
    assert analyse("HALT;;\n").diagnostics == ()


def test_an_unknown_feature_is_refused():
    with pytest.raises(ValueError, match="hover"):
        analyse("HALT;;\n", want={"hover"})


@pytest.mark.parametrize("name", sorted(sources()))
def test_diagnostics_equal_check(name):
    source = sources()[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert list(analyse(source, want=ALL).diagnostics) == diagnostics.check(source)


@pytest.mark.parametrize("path", CORPUS, ids=_id)
def test_diagnostics_equal_check_on_the_kernels(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert list(_analysed(path).diagnostics) == diagnostics.check(path.read_text("utf-8"))


# --- Tokens ---

#: edge_cases.asm, line by line: (line, column, text, role).
EDGE_CASE_TOKENS = {
    12: [(12, 0, "start", "label-def")],
    13: [(13, 4, "SET", "mnemonic"), (13, 8, "lr0", "register"), (13, 12, "cr0", "register")],
    17: [
        (17, 4, "HALT", "mnemonic"),
        (17, 11, "mid_line", "label-def"),
        (17, 21, "HALT", "mnemonic"),
    ],
    21: [(21, 0, "add", "label-def")],
    23: [(23, 0, "lr0", "label-def")],
    28: [(28, 4, "acc.add.first", "mnemonic")],
    30: [
        (30, 4, "MULT.RC.VE", "mnemonic"),
        (30, 15, "lr15", "register"),
        (30, 20, "lr5", "register"),
        (30, 24, "0", "number"),
        (30, 26, "lr15", "register"),
        (30, 31, "cr15", "register"),
    ],
    36: [(36, 4, "ADDBI", "mnemonic"), (36, 10, "lrd2", "register"), (36, 15, "0x1f", "number")],
    39: [
        (39, 4, "BEQ", "mnemonic"),
        (39, 8, "cr0", "register"),
        (39, 12, "cr0", "register"),
        (39, 16, "+1", "number"),
    ],
    46: [
        (46, 4, "BEQ", "mnemonic"),
        (46, 8, "lr0", "register"),
        (46, 12, "lr1", "register"),
        (46, 16, "lr0_scratch", "label-ref"),
    ],
    # `# row {{ 1 + 1 }} of {% if true %}2{% endif %}`: the comment around its Jinja.
    56: [
        (56, 0, "# row ", "comment"),
        (56, 6, "{{ 1 + 1 }}", "jinja"),
        (56, 17, " of ", "comment"),
        (56, 21, "{% if true %}", "jinja"),
        (56, 34, "2", "comment"),
        (56, 35, "{% endif %}", "jinja"),
    ],
    57: [
        (57, 0, "// note ", "comment"),
        (57, 8, '{{ "x" }}', "jinja"),
        (57, 17, " ", "comment"),
        (57, 18, "{# a Jinja comment inside an assembly one #}", "jinja"),
    ],
}


@pytest.mark.parametrize("line", sorted(EDGE_CASE_TOKENS))
def test_edge_case_tokens(line):
    assert _tokens(EDGE_CASES.read_text(encoding="utf-8"), line=line) == EDGE_CASE_TOKENS[line]


def test_a_comment_line_is_one_comment_token():
    assert _tokens("# a comment\n  // another\nHALT;;\n") == [
        (0, 0, "# a comment", "comment"),
        (1, 2, "// another", "comment"),
        (2, 0, "HALT", "mnemonic"),
    ]


def test_a_token_from_an_expression_is_left_to_its_jinja_region():
    source = '{%- set r = "lr0" -%}\nSET {{ r }} cr0;;\n'
    assert _tokens(source, line=1) == [
        (1, 0, "SET", "mnemonic"),
        (1, 4, "{{ r }}", "jinja"),
        (1, 12, "cr0", "register"),
    ]


def test_a_loop_body_is_listed_once():
    source = "{% for i in range(3) %}\nHALT;;\n{% endfor %}\n"
    assert _tokens(source, line=1) == [(1, 0, "HALT", "mnemonic")]


def test_an_enum_operand_and_a_text_that_does_not_parse():
    roles = [role for *_, role in _tokens("ACTIVATE.QUANTIZE relu CR15;;\n")]
    assert roles == ["mnemonic", "enum", "register"]
    # Mid-edit: the stream still says what each token is.
    assert _tokens("SET lr0 cr0 ;; ;;\nHALT") == [
        (0, 0, "SET", "mnemonic"),
        (0, 4, "lr0", "register"),
        (0, 8, "cr0", "register"),
        (1, 0, "HALT", "mnemonic"),
    ]


def _parse_tree_roles(source: str) -> list[tuple[int, int, str, str]]:
    """(line, column, text, LABEL / MNEMONIC / OPERAND / COMMENT) from the parse tree."""
    roles = token_roles(parse_tree(source))
    out = []
    for token in get_parser().lex(source, dont_ignore=True):
        role = roles.get(token.start_pos) or ("COMMENT" if "COMMENT" in token.type else None)
        if role is not None:
            out.append((token.line - 1, token.column - 1, str(token), role))
    return out


_ROLE_CLASS = {"label-def": "LABEL", "mnemonic": "MNEMONIC", "comment": "COMMENT"}


@pytest.mark.parametrize(
    "path", [p for p in CORPUS if not template.has_markers(p.read_text("utf-8"))], ids=_id
)
def test_tokens_of_a_plain_source_agree_with_the_parse_tree(path):
    source = path.read_text(encoding="utf-8")
    got = [
        (t.line, t.col, _text(source, t), _ROLE_CLASS.get(t.role, "OPERAND"))
        for t in _analysed(path).tokens
    ]
    assert got == _parse_tree_roles(source)


_MNEMONICS = frozenset(
    name.upper()
    for name in [i.mnemonic for slot in isa().spec_slot_order for i in isa().instructions(slot)]
    + [p.mnemonic for p in isa().pseudos()]
)
_TOKEN = re.compile(r"[+-]?[A-Za-z0-9_][A-Za-z0-9_.]*")


@pytest.mark.parametrize("path", CORPUS, ids=_id)
def test_tokens_are_in_the_source(path):
    """Disjoint, sorted, one line each; outside Jinja, a line's tokens are what lexing it alone gives."""
    source = path.read_text(encoding="utf-8")
    lines = source.split("\n")
    analysis = _analysed(path)
    jinja_lines = {n for r in analysis.jinja for n in range(r.range.line, r.range.end_line + 1)}
    by_line: dict[int, list[tuple[int, str]]] = {}
    previous = (-1, 0)
    for token in analysis.tokens:
        assert (token.line, token.col) >= previous, "sorted and disjoint"
        assert token.col + token.length <= len(lines[token.line])
        previous = (token.line, token.col + token.length)
        text = _text(source, token)
        if token.role == "mnemonic":
            assert text.upper() in _MNEMONICS
        if token.role not in ("jinja", "comment"):
            assert _TOKEN.fullmatch(text)
            by_line.setdefault(token.line, []).append((token.col, text))
    for number, found in by_line.items():
        if number not in jinja_lines:
            alone = [
                (t.column - 1, str(t)) for t in get_parser().lex(lines[number]) if t.type == "TOKEN"
            ]
            assert found == alone, f"line {number + 1}"


@pytest.mark.parametrize("path", CORPUS, ids=_id)
def test_jinja_tokens_cover_each_region(path):
    analysis = _analysed(path)
    covered = sum(t.length for t in analysis.tokens if t.role == "jinja")
    text = path.read_text(encoding="utf-8")
    expected = sum(
        len(piece) for r in analysis.jinja for piece in text[r.start : r.end].split("\n")
    )
    assert covered == expected


# --- Words ---


def _fits(source: str) -> list[tuple]:
    return [
        (w.range.line, w.mnemonics, dict(w.slots), w.fits, w.error)
        for w in analyse(source, want={"words"}).words
    ]


def test_a_word_that_does_not_fit_hides_no_other():
    source = "SET lr0 cr0;;\nHALT; HALT;;\nHALT;;\n"
    assert _fits(source) == [
        (0, ("SET",), {"lr": 1}, True, None),
        (1, ("HALT", "HALT"), {"cond": 2}, False, "Too many instructions of type CondInst (max 1)"),
        (2, ("HALT",), {"cond": 1}, True, None),
    ]


def test_an_unknown_mnemonic_and_a_bad_pseudo_are_their_words_own():
    source = "FOO lr0;;\nB;;\nHALT;;\n"
    fits = _fits(source)
    assert [fit[3] for fit in fits] == [False, False, True]
    assert fits[0][4] == "Opcode 'FOO' not found in any Inst subclass."
    assert fits[1][4].startswith("Pseudo-instruction 'B' expects 1 operand(s)")


def test_two_words_on_a_line_and_a_loop():
    words = analyse("HALT;; mid: HALT;;\n", want={"words"}).words
    assert [w.range for w in words] == [Range(0, 0, 0, 4), Range(0, 7, 0, 16)]
    loop = "{% for i in range(4) %}\nHALT;;\n{% endfor %}\n"
    assert _fits(loop) == [(1, ("HALT",), {"cond": 1}, True, None)]


def test_a_source_that_does_not_parse_has_no_words():
    assert analyse("SET lr0 ;; ;;\n", want={"words"}).words == ()


def test_edge_case_words():
    words = analyse(EDGE_CASES.read_text(encoding="utf-8"), want={"words"}).words
    assert all(w.fits for w in words)
    assert [w.range.line for w in words][:4] == [12, 17, 17, 21]
    assert words[7].mnemonics == ("MULT.RC.VE", "acc.sub")
    assert dict(words[7].slots) == {"mult": 1, "acc": 1}


@pytest.mark.parametrize("path", CORPUS, ids=_id)
def test_every_word_of_an_assembling_kernel_fits(path):
    analysis = _analysed(path)
    if analysis.diagnostics:
        pytest.skip("does not assemble")
    assert analysis.words and all(w.fits for w in analysis.words)
    source = path.read_text(encoding="utf-8")
    if not template.has_markers(source):
        assert len(analysis.words) == len(AssemblySession(source).parse())


# --- Jinja ---


def _names(source: str) -> list[tuple]:
    return [
        (r.kind, source[r.start : r.end], r.defines, r.uses)
        for r in analyse(source, want={"jinja"}).jinja
    ]


@pytest.mark.parametrize(
    "tag,defines,uses",
    [
        ("{% set a, b = 1, 2 %}", ("a", "b"), ()),
        ("{%- set n = n + step -%}", ("n",), ("n", "step")),
        ("{% for x, y in items if x.lr_row %}", ("x", "y"), ("items", "x")),
        ("{% macro m(r, s=d) %}", ("m", "r", "s"), ("d",)),
        ("{% set q %}", ("q",), ()),
        ("{{ m(lr_row) | upper }}", (), ("m", "lr_row")),
        ('{{ "lr_row" }}', (), ()),
        ("{% elif c > limit %}", (), ("c", "limit")),
        ("{% else %}", (), ()),
        ("{% endfor %}", (), ()),
        ('{% from "lib.j2" import a, b as c %}', ("a", "c"), ()),
        ('{% import "lib.j2" as lib %}', ("lib",), ()),
        ("{% set ns.total = 0 %}", (), ("ns",)),
    ],
)
def test_names_a_tag_defines_and_uses(tag, defines, uses):
    assert _names(tag) == [("block" if tag.startswith("{%") else "variable", tag, defines, uses)]


def test_the_params_header_defines_its_parameters():
    source = "{#- ipu:params\nrows: int = 8\ncpr:  int = 3   # chunks\n-#}\n{# other #}\nHALT;;\n"
    assert [r[2] for r in _names(source)] == [("rows", "cpr"), ()]


def test_region_edges():
    # A closer inside a string, a raw block, and an unclosed tag at the end.
    source = '{{ "}}" }} {% raw %}{{ x }}{% endraw %} SET lr0 {{ lr'
    regions = analyse(source, want={"jinja"}).jinja
    assert [(r.kind, source[r.start : r.end]) for r in regions] == [
        ("variable", '{{ "}}" }}'),
        ("block", "{% raw %}"),
        ("block", "{% endraw %}"),
        ("variable", "{{ lr"),
    ]
    assert regions[-1].range == Range(0, len(source) - 5, 0, len(source))


def test_edge_case_regions():
    regions = analyse(EDGE_CASES.read_text(encoding="utf-8"), want={"jinja"}).jinja
    assert [(r.kind, r.range.line, r.range.column) for r in regions] == [
        ("variable", 56, 6),
        ("block", 56, 21),
        ("block", 56, 35),
        ("variable", 57, 8),
        ("comment", 57, 18),
    ]
    assert regions[0] == JinjaRegion(
        "variable", Range(56, 6, 56, 17), (), (), regions[0].start, regions[0].start + 11
    )


_OPEN = re.compile(r"\{#|\{%|\{\{")
_CLOSE = {"{#": ("comment", "#}"), "{%": ("block", "%}"), "{{": ("variable", "}}")}


def _scanned(raw: str) -> list[dict]:
    """Every tag by a plain scan for its delimiters (right where no string holds a closer)."""
    regions, at = [], 0
    while match := _OPEN.search(raw, at):
        kind, closer = _CLOSE[match.group()]
        end = raw.find(closer, match.end()) + len(closer)
        regions.append({"kind": kind, "start": match.start(), "end": end})
        at = end
    return regions


@pytest.mark.parametrize("path", CORPUS, ids=_id)
def test_kernel_regions_and_names(path):
    raw = path.read_text(encoding="utf-8")
    assert jinja_regions(raw) == _scanned(raw)
    regions = _analysed(path).jinja
    defined = {name for region in regions for name in region.defines}
    used = {name for region in regions for name in region.uses}
    assert used - defined <= {"range", "loop"}


# --- The CLI (check --json --json-version 2) ---


def _check(source: str, *options: str):
    return CliRunner().invoke(cli, ["check", "--json", *options, "--input", "-"], input=source)


def _validate_v2(output: object) -> None:
    schemas = {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in SCHEMAS.glob("check-json-v*.schema.json")
    }
    registry = Registry().with_resources(
        (name, Resource.from_contents(schema)) for name, schema in schemas.items()
    )
    schema = schemas["check-json-v2.schema.json"]
    jsonschema.validators.validator_for(schema)(schema, registry=registry).validate(output)


ANALYSES = ("--tokens", "--words", "--jinja")


@pytest.mark.parametrize("name", sorted(sources()))
def test_v2_with_every_analysis_matches_its_schema_and_v1(name):
    source = sources()[name]
    v1 = _check(source)
    v2 = _check(source, "--json-version", "2", *ANALYSES)
    output = json.loads(v2.stdout)
    _validate_v2(output)
    assert output["diagnostics"] == json.loads(v1.stdout)
    assert v2.exit_code == v1.exit_code
    assert set(output) == {"schemaVersion", "diagnostics", "tokens", "words", "jinja"}


@pytest.mark.parametrize("path", [CORPUS[0], max(CORPUS, key=lambda p: p.stat().st_size)], ids=_id)
def test_v2_of_a_kernel_matches_its_schema(path):
    output = json.loads(_check(path.read_text("utf-8"), "--json-version", "2", *ANALYSES).stdout)
    _validate_v2(output)
    assert output["tokens"] and output["words"]


@pytest.mark.parametrize("option", ANALYSES)
def test_each_option_adds_its_own_key(option):
    output = json.loads(_check("{% set x = 1 %}HALT;;\n", "--json-version", "2", option).stdout)
    assert set(output) == {"schemaVersion", "diagnostics", option.removeprefix("--")}


@pytest.mark.parametrize(
    "options",
    [
        ["--json", "--tokens"],
        ["--json", "--json-version", "1", "--words"],
        ["--text", "--jinja"],
        ["--json-version", "2", "--tokens"],
    ],
)
def test_an_analysis_outside_v2_is_a_usage_error(options):
    result = CliRunner().invoke(cli, ["check", *options, "--input", "-"], input="HALT;;\n")
    assert result.exit_code == 2
    assert "Usage:" in result.output


# --- Performance ---


def test_analysing_the_largest_kernel_is_fast(record_property):
    path = max(sorted(KERNELS.rglob("*.asm")), key=lambda p: p.stat().st_size)
    source = path.read_text(encoding="utf-8")
    analyse("start: HALT;;\n{{ 1 }}\n", want=ALL)  # build the parser and the ISA tables once
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        start = time.perf_counter()
        analyse(source, want=ALL)
        first = (time.perf_counter() - start) * 1000
    record_property("analyse_ms", round(first, 1))
    print(f"analyse({path.name}, all features): {first:.0f} ms (budget {BUDGET_MS} ms)")
    assert first < HARD_LIMIT_MS
