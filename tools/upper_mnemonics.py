"""Upper-case the mnemonics and register names of assembly sources (E07-01, Q45).

    bazel run //tools:upper_mnemonics -- [--check] FILE...

CLAUDE.md writes mnemonics and register names in upper case; the assembler
accepts any case, so the rewrite is binary-identical (its test holds every
program of the E02 asm corpus to its digest). ``//tools:test_upper_mnemonics``
runs ``--check`` over every ``.asm`` and ``.asm.j2`` of ``kernels/``, so a
kernel written in lower case fails there; run this on it to fix it.

The source is read the way the assembler reads it: Jinja tags come from
Jinja's own lexer (``ipu_as.jinja_tags``), and the text between them is cut
into the grammar's own terminals (``ipu_as.syntax.get_parser``). Then:

* an instruction's first token (at the start of the source, or after ``;``,
  ``;;`` or a label) that names an instruction or pseudo-instruction becomes
  upper case (``ld.mult`` -> ``LD.MULT``);
* an operand that names a register becomes upper case (``lr0`` -> ``LR0``,
  ``cr15`` -> ``CR15``).

Nothing else changes: Jinja tags (``{{ }}``, ``{% %}``, ``{# #}``) and what is
inside them, comments, labels (a label named like a register keeps its
spelling everywhere), numbers, enum operand values (``identity`` in
``ACTIVATE.QUANTIZE identity CR15``), and a token that touches a tag with no
space between (``lr{{ i }}``), which renders into a different token. After a
``{{ }}`` tag the next token may be an operand, so a mnemonic there is
upper-cased only when it starts its line.

Prints one line per file that changes, with the number of tokens. ``--check``
writes nothing and exits 1 if any file would change.
"""

from __future__ import annotations

import argparse
import functools
import os
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import ipu_as.template as template
from ipu_as.encode import WILDCARD_OPCODES
from ipu_as.jinja_tags import tag_spans
from ipu_as.syntax import get_parser
from ipu_isa.operand_types import OPERAND_TYPES, OperandKind
from ipu_isa.slots import SLOTS
from ipu_isa.view import isa

__all__ = ["main", "upper_text"]

#: The grammar terminals the rewrite tells apart, in the order they are tried.
_ORDER = ("COMMENT_HASH", "COMMENT_LINE", "_SEMI2", "_SEMI", "COLON", "TOKEN", "_NL", "WS_INLINE")
_COMMENTS = ("COMMENT_HASH", "COMMENT_LINE")
_SPACE = ("_NL", "WS_INLINE")


@functools.cache
def _scanner() -> re.Pattern[str]:
    terminals = {t.name: t.pattern.to_regexp() for t in get_parser().terminals}
    return re.compile(
        "|".join(f"(?P<{name}>{terminals[name]})" for name in _ORDER) + r"|(?P<JUNK>.)",
        re.DOTALL,
    )


@functools.cache
def mnemonics() -> frozenset[str]:
    """Every instruction and pseudo-instruction mnemonic, upper case."""
    names = {i.mnemonic.upper() for slot in SLOTS for i in isa().instructions(slot)}
    names |= {p.mnemonic.upper() for p in isa().pseudos()}
    return frozenset(names | {w.upper() for w in WILDCARD_OPCODES})


@functools.cache
def registers() -> frozenset[str]:
    """Every register name an operand can hold, upper case."""
    return frozenset(
        member.upper()
        for operand_type in OPERAND_TYPES.values()
        if operand_type.kind is OperandKind.REGISTER
        for member in operand_type.domain.members
    )


@dataclass(frozen=True)
class _Item:
    """One piece of the source: a Jinja tag (kind ``jinja-<kind>``) or a lexeme."""

    kind: str
    start: int
    end: int


def _items(text: str) -> Iterator[_Item]:
    """The source as Jinja tags and grammar lexemes, in order, covering all of it.

    A comment that a tag interrupts carries on after the tag to the end of its
    line, as the rendered text reads it.
    """
    spans = tag_spans(text) if template.has_markers(text) else ()
    at = 0
    in_comment = False
    for span in (*spans, None):
        stop = len(text) if span is None else span.start
        position = at
        if in_comment:  # the rest of a comment that a tag interrupted
            newline = text.find("\n", at, stop)
            position = stop if newline < 0 else newline
            if position > at:
                yield _Item("COMMENT_LINE", at, position)
            in_comment = newline < 0
        for match in _scanner().finditer(text, position, stop):
            kind = match.lastgroup or "JUNK"
            yield _Item(kind, match.start(), match.end())
            if kind in _COMMENTS and match.end() == stop and span is not None:
                in_comment = True
        if span is not None:
            yield _Item(f"jinja-{span.kind}", span.start, span.end)
            at = span.end


def _labels(text: str, items: list[_Item]) -> frozenset[str]:
    """The names the source defines as labels (a token, then ``:``), upper case."""
    code = [i for i in items if i.kind not in _SPACE and i.kind not in _COMMENTS]
    return frozenset(
        text[a.start : a.end].upper()
        for a, b in zip(code, code[1:], strict=False)
        if a.kind == "TOKEN" and b.kind == "COLON"
    )


def _touches_tag(items: list[_Item], index: int) -> bool:
    """Whether a tag sits right before or after item ``index``, with no space between."""
    before = items[index - 1] if index else None
    after = items[index + 1] if index + 1 < len(items) else None
    return any(item is not None and item.kind.startswith("jinja-") for item in (before, after))


def _rewritten(upper: str, position: str, line_start: bool) -> bool:
    """Whether a token (upper case) is a mnemonic or register name to upper-case.

    ``position`` is ``start`` where an instruction begins, ``operand`` after
    its mnemonic, and ``unknown`` after a ``{{ }}`` tag (which may have rendered
    either).
    """
    if upper in mnemonics():
        return position == "start" or (position == "unknown" and line_start)
    return upper in registers() and position != "start"


#: Items after which the next token begins an instruction.
_STARTS = ("_SEMI", "_SEMI2", "COLON")
#: Items that change nothing about where the next token is.
_NEUTRAL = ("WS_INLINE", *_COMMENTS, "jinja-block", "jinja-comment")


def upper_text(text: str) -> tuple[str, int]:
    """Upper-case the mnemonics and register names of one source.

    Args:
        text: An assembly source or template.

    Returns:
        The new text and the number of tokens changed.
    """
    items = list(_items(text))
    labels = _labels(text, items)
    changes: list[tuple[int, int, str]] = []
    position, line_start = "start", True
    for index, item in enumerate(items):
        if item.kind == "_NL":
            line_start = True
            continue
        if item.kind in _NEUTRAL:
            continue
        if item.kind != "TOKEN":
            position = (
                "start"
                if item.kind in _STARTS
                else "unknown"
                if item.kind == "jinja-variable"
                else "operand"
            )
            line_start = False
            continue
        word = text[item.start : item.end]
        after = items[index + 1] if index + 1 < len(items) else None
        is_label = after is not None and after.kind == "COLON"
        upper = word.upper()
        if (
            word != upper
            and not is_label
            and upper not in labels
            and not _touches_tag(items, index)
            and _rewritten(upper, position, line_start)
        ):
            changes.append((item.start, item.end, upper))
        if not is_label:
            position = "operand"
        line_start = False
    out = []
    at = 0
    for start, end, new in changes:
        out.append(text[at:start])
        out.append(new)
        at = end
    out.append(text[at:])
    return "".join(out), len(changes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="+", type=Path, help="the sources to rewrite")
    parser.add_argument("--check", action="store_true", help="write nothing; exit 1 on changes")
    args = parser.parse_args(argv)
    base = Path(os.environ.get("BUILD_WORKING_DIRECTORY", "."))
    changed = 0
    for file in args.files:
        path = file if file.is_absolute() else base / file
        text = path.read_text(encoding="utf-8")
        new, count = upper_text(text)
        if count:
            changed += 1
            print(f"{count:6d}  {file}")
            if not args.check:
                path.write_text(new, encoding="utf-8")
    verb = "would change" if args.check else "changed"
    print(f"{changed} file(s) {verb}")
    return 1 if args.check and changed else 0


if __name__ == "__main__":
    sys.exit(main())
