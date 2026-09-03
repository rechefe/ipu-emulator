#!/usr/bin/env python3
"""Generate the VS Code TextMate grammar for IPU assembly.

The grammar is built from two sources, and it is worth being precise about
which is which:

**Structure** — comments, `;`/`;;`, `:`, and the identifier shape — comes from
``get_parser().terminals`` via ``pattern.to_regexp()``. Nothing here restates a
token's text, so changing a terminal in ``asm_grammar.lark`` reaches the editor
on the next build.

**Vocabulary** — which identifiers are instructions and which are registers —
comes from ``instruction_spec`` and the register enums. It cannot come from the
parser: ``asm_grammar.lark`` has a single catch-all ``TOKEN`` terminal, so the
parser genuinely does not distinguish ``BKPT`` from ``lr0`` from ``my_label``.
Both layers are enumerated rather than listed by hand, so adding an instruction
to the spec also reaches the editor on the next build.

The one thing neither layer supplies is which *colour* a terminal should get: a
parser knows ``COMMENT_LINE`` exists, not that it should render as a comment.
That mapping lives in :data:`TERMINAL_SCOPES` / :data:`SCOPE_CONVENTIONS`, and
an unmapped terminal is a **hard error** rather than a silent omission — so
adding a terminal to the grammar fails the build at the point of the change
instead of quietly going unhighlighted.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path

from ipu_common.instruction_spec import (
    INSTRUCTION_SPEC,
    PSEUDO_INSTRUCTION_SPEC,
    SLOT_METADATA,
)
from ipu_common.registers import create_assembler_reg_enums

from ipu_as.lark_tree import get_parser
from ipu_as.reg import LRD_REG_FIELDS

GENERATED_BY = "//vscode-ipu-asm:gen_vscode from asm_grammar.lark — do not edit by hand"

SCOPE_NAME = "source.ipu-asm"

#: Characters that may continue an identifier. A keyword pattern must not match
#: when one of these follows, or it bites a prefix out of a longer name — `BNE`
#: out of the label `bne_target`. `.` is included because mnemonics contain
#: dots (`ACC.ADD.FIRST`), which makes `\b` unusable here.
_IDENT_CONTINUE = r"[A-Za-z0-9_.]"

#: Mirrors the assembler's int(value, 0) handling in ipu_token, plus the leading
#: sign a relative branch target (`+2`) uses.
_NUMBER_PATTERN = (
    r"[+-]?(?:0[xX][0-9a-fA-F_]+|0[bB][01_]+|0[oO][0-7_]+|\d+)"
    rf"(?!{_IDENT_CONTINUE})"
)

#: Terminal name -> TextMate scope. ``None`` means "the parser matches it, but
#: it carries no colour" (whitespace), which is different from "unknown".
TERMINAL_SCOPES: dict[str, str | None] = {
    "TOKEN": "variable.other.ipu-asm",
    "COLON": "punctuation.separator.label.ipu-asm",
    "_SEMI2": "punctuation.terminator.bundle.ipu-asm",
    "_SEMI": "punctuation.separator.slot.ipu-asm",
    "COMMENT_LINE": "comment.line.double-slash.ipu-asm",
    "COMMENT_HASH": "comment.line.number-sign.ipu-asm",
    "WS_INLINE": None,
    "_NL": None,
}

#: Fallback for terminals added later that follow the naming convention, so the
#: common cases need no edit here at all.
SCOPE_CONVENTIONS: tuple[tuple[str, str | None], ...] = (
    ("COMMENT_", "comment.block.ipu-asm"),
    ("STRING_", "string.quoted.ipu-asm"),
    ("PUNCT_", "punctuation.other.ipu-asm"),
    ("WS_", None),
)

#: Terminals whose match is more specific than a bare identifier, so they must
#: be tried first; otherwise TOKEN shadows everything.
_TERMINAL_PRECEDENCE = ("COMMENT_LINE", "COMMENT_HASH", "_SEMI2", "_SEMI", "COLON", "TOKEN")

_JINJA_PUNCTUATION = "punctuation.definition.template-expression.jinja.ipu-asm"

#: Jinja2 control words. Unlike the ISA vocabulary these are hand-listed, and
#: legitimately so: Jinja is a fixed external language the parser never sees, so
#: there is nothing in this repo to derive them from.
_JINJA_KEYWORDS = (
    "set", "for", "endfor", "if", "elif", "else", "endif", "macro", "endmacro",
    "call", "endcall", "filter", "endfilter", "include", "import", "from",
    "extends", "block", "endblock", "with", "endwith", "raw", "endraw", "do",
    "in", "is", "not", "and", "or", "true", "false", "none",
)

#: Shared by `{% … %}` and `{{ … }}`; without these the whole construct carries
#: only a `meta.*` scope, which themes do not style — so it looks unhighlighted.
_JINJA_INNER_PATTERNS = [
    {
        "name": "keyword.control.jinja.ipu-asm",
        "match": rf"\b(?:{'|'.join(_JINJA_KEYWORDS)})\b",
    },
    {"name": "string.quoted.double.jinja.ipu-asm", "match": r"\"[^\"]*\""},
    {"name": "string.quoted.single.jinja.ipu-asm", "match": r"'[^']*'"},
    {"name": "constant.numeric.jinja.ipu-asm", "match": r"\b\d+\b"},
    # `|` is Jinja's filter operator, so it belongs here rather than with the
    # arithmetic it sits beside.
    {"name": "keyword.operator.jinja.ipu-asm", "match": r"[=+\-*/%<>!~|]+"},
    {"name": "variable.other.jinja.ipu-asm", "match": r"\b[A-Za-z_][A-Za-z0-9_]*\b"},
]


class UnmappedTerminalError(RuntimeError):
    """Raised when the grammar grew a terminal with no known TextMate scope."""


def scope_for(name: str) -> str | None:
    """Resolve a terminal name to a TextMate scope, or raise."""
    if name in TERMINAL_SCOPES:
        return TERMINAL_SCOPES[name]
    for prefix, scope in SCOPE_CONVENTIONS:
        if name.startswith(prefix):
            return scope
    raise UnmappedTerminalError(
        f"Terminal {name!r} in asm_grammar.lark has no TextMate scope.\n"
        f"Add it to TERMINAL_SCOPES in {__name__}, or rename it with one of the "
        f"conventional prefixes: {', '.join(p for p, _ in SCOPE_CONVENTIONS)}."
    )


# ---------------------------------------------------------------------------
# Vocabulary layer
# ---------------------------------------------------------------------------

def _mnemonics() -> list[str]:
    real = {name for slot in INSTRUCTION_SPEC.values() for name in slot}
    return sorted(real | set(PSEUDO_INSTRUCTION_SPEC))


def _registers() -> list[str]:
    names = {v for values in create_assembler_reg_enums().values() for v in values}
    names.update(LRD_REG_FIELDS)
    return sorted(names)


def _keyword_pattern(values) -> str:
    """Case-insensitive whole-token alternation, longest match first.

    Longest-first keeps `ACC.ADD` from shadowing `ACC.ADD.FIRST`. The trailing
    lookahead is what stops `BNE` matching inside `bne_target`.
    Case-insensitivity mirrors the assembler, which lowercases in
    `Opcode.find_opcode_class`.
    """
    ordered = sorted(set(values), key=lambda v: (-len(v), v))
    alternation = "|".join(re.escape(v) for v in ordered)
    return f"(?i:{alternation})(?!{_IDENT_CONTINUE})"


# ---------------------------------------------------------------------------
# Structure layer
# ---------------------------------------------------------------------------

def _label_pattern() -> dict:
    """Derive the label rule from the grammar's own ``label`` expansions.

    A label name is lexically just an identifier — `input_loop` is a `TOKEN`
    exactly like a branch target is — so the only thing distinguishing it is the
    trailing `:`. TextMate has no parser, so this positional rule is how it
    recovers what the grammar knows structurally.

    Built from ``parser.rules`` rather than written by hand, so changing the
    label syntax carries through automatically.
    """
    parser = get_parser()
    by_name = {t.name: t for t in parser.terminals}

    leading: dict[str, list[str]] = {}
    for rule in parser.rules:
        if rule.origin.name != "label":
            continue
        symbols = [s.name for s in rule.expansion]
        if len(symbols) != 2 or not all(s in by_name for s in symbols):
            raise UnmappedTerminalError(
                f"The 'label' rule has an unexpected shape {symbols!r}; "
                f"gen_vscode can only derive a <name> <terminator> label. "
                f"Update _label_pattern() to match the new grammar."
            )
        name_term, terminator = symbols
        leading.setdefault(terminator, []).append(name_term)

    if len(leading) != 1:
        raise UnmappedTerminalError(
            f"The 'label' rule uses multiple terminators {sorted(leading)}; "
            f"update _label_pattern()."
        )

    terminator, name_terms = next(iter(leading.items()))
    names = "|".join(by_name[n].pattern.to_regexp() for n in sorted(set(name_terms)))
    # \s* stands in for the whitespace the grammar %ignores between the two
    # terminals (WS_INLINE and _NL are both ignored, and \s covers both).
    return {
        "match": f"({names})\\s*({by_name[terminator].pattern.to_regexp()})",
        "captures": {
            "1": {"name": "entity.name.label.ipu-asm"},
            "2": {"name": scope_for(terminator)},
        },
    }


def _ordered_terminals():
    """Terminals in match-precedence order, most specific first."""
    by_name = {t.name: t for t in get_parser().terminals}
    for name in _TERMINAL_PRECEDENCE:
        if name in by_name:
            yield by_name.pop(name)
    # Anything the grammar gained since: deterministic order, and scope_for()
    # rejects it if it is not mappable.
    for name in sorted(by_name):
        yield by_name[name]


def build_tmlanguage() -> dict:
    """Assemble the TextMate grammar from the parser plus the ISA vocabulary."""
    repository: dict[str, dict] = {}
    terminal_includes = []

    for terminal in _ordered_terminals():
        scope = scope_for(terminal.name)
        if scope is None:
            continue  # matched by the parser, but nothing to colour
        key = terminal.name.lstrip("_").lower().replace("_", "-")
        repository[key] = {
            "name": scope,
            # Straight from the parser. Oniguruma accepts Lark's Python-flavour
            # output for these patterns: character classes, non-capturing
            # groups, scoped (?i:…) and lookahead are common to both engines.
            "match": terminal.pattern.to_regexp(),
        }
        terminal_includes.append({"include": f"#{key}"})

    # Vocabulary. The parser cannot supply these — every one of them is a TOKEN
    # to it — so they are enumerated from the spec instead.
    repository["mnemonic"] = {
        "name": "keyword.other.mnemonic.ipu-asm",
        "match": _keyword_pattern(_mnemonics()),
    }
    repository["register"] = {
        "name": "variable.language.register.ipu-asm",
        "match": _keyword_pattern(_registers()),
    }
    repository["number"] = {
        "name": "constant.numeric.ipu-asm",
        "match": _NUMBER_PATTERN,
    }

    # Jinja2 is a preprocessing layer the parser never sees: lark_tree renders
    # it away before parsing. It cannot come from parser.terminals, so it is
    # declared here and must lead, or a `{#- … -#}` header is shredded by the
    # COMMENT_HASH rule.
    #
    # The inner patterns matter: `meta.*` is not styled by themes, so without
    # them a whole `{%- set lr_off = "lr0" -%}` renders as dead plain text even
    # though it was "matched". Register aliasing like that is how every kernel
    # in this repo names its registers, so it is most of what an author looks at.
    repository["jinja-comment"] = {
        "name": "comment.block.jinja.ipu-asm",
        "begin": r"\{#-?",
        "end": r"-?#\}",
    }
    repository["jinja-statement"] = {
        "name": "meta.embedded.block.jinja.ipu-asm",
        "begin": r"\{%-?",
        "end": r"-?%\}",
        "beginCaptures": {"0": {"name": _JINJA_PUNCTUATION}},
        "endCaptures": {"0": {"name": _JINJA_PUNCTUATION}},
        "patterns": _JINJA_INNER_PATTERNS,
    }
    repository["jinja-expression"] = {
        "name": "meta.embedded.line.jinja.ipu-asm",
        "begin": r"\{\{-?",
        "end": r"-?\}\}",
        "beginCaptures": {"0": {"name": _JINJA_PUNCTUATION}},
        "endCaptures": {"0": {"name": _JINJA_PUNCTUATION}},
        "patterns": _JINJA_INNER_PATTERNS,
    }

    # Order matters. Jinja first, then labels (a label may be spelled like a
    # mnemonic — `add:` — and whichever rule matches first wins), then the
    # vocabulary, then the bare terminals with TOKEN last as the catch-all.
    repository["label"] = _label_pattern()
    patterns = (
        [
            {"include": "#jinja-comment"},
            {"include": "#jinja-statement"},
            {"include": "#jinja-expression"},
            {"include": "#label"},
        ]
        + [i for i in terminal_includes if i["include"].startswith("#comment")]
        + [
            {"include": "#mnemonic"},
            {"include": "#register"},
            {"include": "#number"},
        ]
        + [i for i in terminal_includes if not i["include"].startswith("#comment")]
    )

    return {
        "$schema": (
            "https://raw.githubusercontent.com/martinring/tmlanguage/master/tmlanguage.json"
        ),
        "name": "IPU Assembly",
        "scopeName": SCOPE_NAME,
        "comment": f"Generated by {GENERATED_BY}",
        "patterns": patterns,
        "repository": repository,
    }


def build_language_configuration() -> dict:
    """Editor ergonomics: comment toggling, brackets, word selection.

    ``wordPattern`` is taken from the parser's own ``TOKEN`` terminal, which is
    what makes double-click and hover select ``ACC.ADD.FIRST`` whole instead of
    stopping at the first dot.

    The comment *markers* cannot be derived: the grammar defines comments as
    regexes (``\\/\\/[^\\n]*``), and recovering the literal ``//`` an editor
    needs for toggling would mean inverting a regex. They are stated here, and
    the agreement test keeps them honest — it checks that text these markers
    introduce really is scoped as a comment by the generated grammar.
    """
    token = {t.name: t for t in get_parser().terminals}["TOKEN"]
    return {
        "comments": {"lineComment": "//", "blockComment": ["{#", "#}"]},
        "brackets": [["{%", "%}"], ["{{", "}}"], ["{#", "#}"]],
        "autoClosingPairs": [
            {"open": "{%", "close": " %}"},
            {"open": "{{", "close": " }}"},
            {"open": "{#", "close": " #}"},
        ],
        "wordPattern": token.pattern.to_regexp(),
    }


#: Fields of InstructionDoc that hold assembly code rather than prose.
_ASM_SNIPPET_FIELDS = ("syntax", "example")


def _normalize_asm_snippet(text: str) -> str:
    """Rewrite comma-separated operands to the whitespace form that assembles.

    `asm_grammar.lark` has no comma terminal, so `MULT.RC.VE LR0, LR3, 0, LR2,
    CR15;;` — copied verbatim from that instruction's own `example` — is a lex
    error. Every InstructionDoc writes operands with commas anyway, following
    the prose convention in CLAUDE.md rather than the grammar.

    A hover must not teach a syntax that fails to assemble, so the two code
    fields are normalized to match the grammar; prose keeps its commas. The
    underlying docs/grammar disagreement is pre-existing; `test_hover_data.py`
    asserts every shipped example really assembles, so this cannot rot.
    """
    return re.sub(r",\s*", " ", text)


def _doc_to_dict(doc) -> dict | None:
    if doc is None:
        return None
    data = asdict(doc) if is_dataclass(doc) else dict(doc)
    for field in _ASM_SNIPPET_FIELDS:
        if isinstance(data.get(field), str):
            data[field] = _normalize_asm_snippet(data[field])
    return data


def build_hover_data() -> dict:
    """Instruction and register reference, consumed by the extension at runtime.

    A mnemonic can occupy more than one slot — NOP exists in all nine, each with
    its own summary — so each entry keeps every form rather than flattening.
    """
    instructions: dict[str, list[dict]] = {}

    for slot, entries in INSTRUCTION_SPEC.items():
        for opcode, (mnemonic, entry) in enumerate(entries.items()):
            instructions.setdefault(mnemonic, []).append(
                {
                    "slot": slot,
                    "opcode": opcode,
                    "pseudo": False,
                    "operands": [dict(op) for op in entry.get("operands", [])],
                    "doc": _doc_to_dict(entry.get("doc")),
                }
            )

    for mnemonic, entry in PSEUDO_INSTRUCTION_SPEC.items():
        expands_to = entry.get("expands_to", {})
        instructions.setdefault(mnemonic, []).append(
            {
                "slot": expands_to.get("slot", "cond"),
                "pseudo": True,
                "expandsTo": expands_to.get("instruction"),
                "operands": [dict(op) for op in entry.get("operands", [])],
                "doc": _doc_to_dict(entry.get("doc")),
            }
        )

    return {
        "_generated": GENERATED_BY,
        "instructions": {
            mnemonic: sorted(forms, key=lambda f: (f["slot"], f.get("opcode", -1)))
            for mnemonic, forms in instructions.items()
        },
        "registers": _registers(),
        "slots": {
            slot: {
                "description": meta.get("description", ""),
                "hardware": meta.get("hardware", True),
            }
            for slot, meta in SLOT_METADATA.items()
        },
    }


def _dumps(obj) -> str:
    """Deterministic JSON: same grammar in, same bytes out."""
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def render_tmlanguage() -> str:
    return _dumps(build_tmlanguage())


def render_language_configuration() -> str:
    return _dumps(build_language_configuration())


def render_hover_data() -> str:
    return _dumps(build_hover_data())


#: Relative path -> renderer, shared by the wrapper and any test.
GENERATED_FILES = {
    "syntaxes/ipu-asm.tmLanguage.json": render_tmlanguage,
    "language-configuration.json": render_language_configuration,
    "data/isa.json": render_hover_data,
}


def generate_all(out_dir: Path) -> None:
    for rel_path, render in GENERATED_FILES.items():
        target = out_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render(), encoding="utf-8")
