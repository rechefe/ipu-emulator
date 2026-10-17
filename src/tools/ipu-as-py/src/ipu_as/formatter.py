"""The repository's layout for an assembly source, applied by the assembler (`ipu-as format`).

    format_source("loop: SET lr0 cr0;; HALT;;")  # 'loop:\\n    SET lr0 cr0;;\\n    HALT;;'

The rules are the editor's formatter's (vscode-ipu-asm/language.js, `format`),
ported one to one; test_format.py holds this module to that formatter's
recorded output on the extension's corpus and on every kernel:

- a label sits at column 0, on its own line;
- an instruction line is indented one level;
- `;;` ends its line: code after it moves to the next line;
- with `blank_line`, an empty line separates each `;;` word from the next.

Only whitespace between those pieces changes. Lines that are only Jinja or
comments keep their layout, as do lines inside a multi-line Jinja tag and
macro calls that expand to whole words (`{{ block(0) }}`). Trailing
whitespace goes, except inside a tag that continues on the next line.

Like the editor's, the formatter reads the source text itself, with Jinja
tags and comments masked out, not a render: it must format a template it
cannot render (no defines) and a file being edited (does not parse). The
lexical patterns (token, label, comments, `;`, `;;`) are the grammar's own
terminals; only how Jinja delimits a tag is written here.

Before it returns changed text, `format_source` checks that the change is
layout only (`check_same_program`), and raises `FormatError` if it is not.
"""

from __future__ import annotations

import bisect
import functools
import re
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import ipu_as.template as template
from ipu_as.errors import AssemblerError, FormatError
from ipu_as.jinja_tags import tag_spans
from ipu_as.placement import COMMENTS
from ipu_as.session import AssemblySession
from ipu_as.syntax import get_parser

__all__ = ["check_same_program", "format_source"]

# --- Lexical facts ---

#: What JavaScript's `trim` removes: the editor's formatter decides by it.
_JS_SPACE = "\t\n\v\f\r                  　﻿"
#: Where a Jinja tag opens, and how each kind closes (language.js `JINJA_CLOSE`).
_JINJA_OPEN = re.compile(r"\{#|\{%|\{\{")
_JINJA_CLOSE = {"{#": ("comment", "#}"), "{%": ("block", "%}"), "{{": ("variable", "}}")}
#: A `{{ }}` tag that calls a macro: `{{ name(`.
_CALL = re.compile(r"\{\{-?\s*[A-Za-z_][\w.]*\s*\(", re.ASCII)
#: A line that closes a Jinja block, or opens its next branch: it stays with the word before it.
_CLOSER = re.compile(r"[ \t]*\{%-?\s*(end\w*|else|elif)\b", re.ASCII)
_LEADING = re.compile(r"[ \t]*")
_TRAILING = re.compile(r"[ \t]+\Z")
_LABEL_COLON = re.compile(r"[ \t]*:")


@dataclass(frozen=True)
class _Lexicon:
    """The grammar's terminals the formatter needs, as patterns and strings."""

    label: re.Pattern[str]
    comments: re.Pattern[str]
    slot_separator: str
    word_terminator: str


@functools.cache
def _lexicon() -> _Lexicon:
    terminals = {t.name: t.pattern.to_regexp() for t in get_parser().terminals}
    # `\s*` stands for the whitespace the grammar ignores between a label and its `:`.
    label = re.compile(f"({terminals['TOKEN']})\\s*({terminals['COLON']})")
    comments = re.compile("|".join(f"(?:{terminals[name]})" for name in sorted(COMMENTS)))
    return _Lexicon(label, comments, terminals["_SEMI"], terminals["_SEMI2"])


def _strip(text: str) -> str:
    return text.strip(_JS_SPACE)


def _rstrip(text: str) -> str:
    return text.rstrip(_JS_SPACE)


def _leading(text: str) -> int:
    match = _LEADING.match(text)
    return 0 if match is None else match.end()


# --- The masked text: assembly with Jinja and comments taken out ---


@dataclass(frozen=True)
class _Region:
    """One Jinja tag, as offsets; an unclosed tag runs to the end."""

    kind: str
    start: int
    end: int


def _jinja_regions(text: str) -> list[_Region]:
    """Every `{# #}`, `{% %}` and `{{ }}` of `text`, in order (language.js `jinjaRegions`)."""
    regions = []
    at = 0
    while (match := _JINJA_OPEN.search(text, at)) is not None:
        kind, closer = _JINJA_CLOSE[match.group()]
        close = text.find(closer, match.start() + 2)
        end = len(text) if close < 0 else close + len(closer)
        regions.append(_Region(kind, match.start(), end))
        at = end
    return regions


def _blank(span: str, fill: str, across_lines: bool = False) -> str:
    """`span` with each character replaced by `fill`, keeping line breaks unless `across_lines`."""
    if across_lines or "\n" not in span:
        return fill * len(span)
    return re.sub(r"[^\n]", fill, span)


class _Masked:
    """A source as the assembly lexer should see it, offsets preserved (language.js `mask`).

    `{# #}` and `{% %}` become blanks; a `{{ }}` becomes one identifier-shaped
    run (`J`s), since in assembly position it renders to a token; comments
    become blanks. Comments are found last, so a `#` inside a tag starts none.

    Args:
        text: The source.
    """

    def __init__(self, text: str) -> None:
        self.text = text
        self.regions = _jinja_regions(text)
        self.by_start = {region.start: region for region in self.regions}
        self._starts = [region.start for region in self.regions]
        jinja: list[str] = []
        at = 0
        for region in self.regions:
            variable = region.kind == "variable"
            jinja.append(text[at : region.start])
            jinja.append(
                _blank(text[region.start : region.end], "J" if variable else " ", variable)
            )
            at = region.end
        jinja.append(text[at:])
        self.masked = _lexicon().comments.sub(lambda m: _blank(m.group(), " "), "".join(jinja))

    def in_region(self, at: int) -> bool:
        """True when a tag starts before `at` and ends after it (language.js `regionOver`)."""
        k = bisect.bisect_left(self._starts, at)
        return k > 0 and self.regions[k - 1].end > at

    def words_call(self, at: int) -> bool:
        """True when a tag at `at` is a macro call that expands to whole words.

        `{{ block(0) }}` is one; a call followed by `:` names a label instead.
        """
        region = self.by_start.get(at)
        if region is None or region.kind != "variable":
            return False
        if not _CALL.match(self.text, region.start, region.end):
            return False
        return _LABEL_COLON.match(self.masked[region.end : region.end + 80]) is None

    def ends_word(self, masked: str) -> bool:
        return _rstrip(masked).endswith(_lexicon().word_terminator)


# --- Formatting ---


@dataclass(frozen=True)
class _Line:
    """One output line, and its part in the word structure (language.js `emit`)."""

    text: str
    eol: str
    keep: bool
    ends: bool = False
    code: bool = False
    inside: bool = False

    @property
    def empty(self) -> bool:
        return not _strip(self.text)

    @property
    def closer(self) -> bool:
        return not self.code and _CLOSER.match(self.text) is not None


def _gaps(lines: Sequence[_Line]) -> set[int]:
    """Where each word is set apart from the one before it (language.js `gapsIn`).

    For a word with code after it: the first line after it that is not a
    closing Jinja tag (so a comment or `{% for %}` goes with the word it
    introduces). None where an empty line already separates the two, and never
    on a line that continues a tag begun above it.
    """
    out: set[int] = set()
    gap: list[int] | None = None  # None: no word just ended; []: no place found yet
    for i, line in enumerate(lines):
        if gap is not None:
            if line.empty:
                gap = None
            elif line.code:
                out.add(gap[0] if gap else i)
                gap = None
            elif not gap and not line.closer and not line.inside:
                gap.append(i)
        if line.ends:
            gap = []
    return out


@dataclass(frozen=True)
class _Piece:
    """Part of one line's code: from column `at`, up to and including a `;;`."""

    at: int
    text: str


@dataclass(frozen=True)
class _Context:
    """The line a piece is cut from: its text, its masked text, its indent and offset."""

    body: str
    masked: str
    lead: int
    start: int


class _Formatter:
    """One run of `format_source` over a text (language.js `Formatter`).

    Lines are worked on without their `\\r`, and each is written back with
    its own line ending (the pieces of a split line share it), so a CRLF file
    stays CRLF and a mixed one keeps its mix.
    """

    def __init__(self, text: str, indent: str) -> None:
        self.source = _Masked(text)
        self.indent = indent
        self.lines = text.split("\n")
        self.starts = [0]
        for line in self.lines[:-1]:
            self.starts.append(self.starts[-1] + len(line) + 1)
        self.out: list[_Line] = []
        self.keep = True
        self.eol = "\n"
        # Whether the last instruction carries on to the next line (no `;` yet):
        # that line is its continuation and keeps its own deeper indent.
        self.open = False

    def run(self, first: int, last: int, blank_line: bool) -> str:
        """Lines [first, last] formatted, joined by their line endings; no ending after the last."""
        start, stop = self._window(first, last, blank_line)
        for n in range(start, stop + 1):
            self.keep = first <= n <= last
            self._line(n)
        gaps = _gaps(self.out) if blank_line else set()
        kept: list[_Line] = []
        for k, line in enumerate(self.out):
            if line.keep:
                if k in gaps:
                    kept.append(_Line("", line.eol, True))
                kept.append(line)
        return "".join(
            line.text + (line.eol if i < len(kept) - 1 else "") for i, line in enumerate(kept)
        )

    def _window(self, first: int, last: int, blank_line: bool) -> tuple[int, int]:
        """The lines to work on: from the last code line before the range, which says
        whether an instruction carries on into it (all lines, to place empty lines)."""
        lines, starts, masked = self.lines, self.starts, self.source.masked
        if blank_line:
            return 0, len(lines) - 1
        n = min(first, len(lines) - 1)
        while n > 0 and not (
            _strip(masked[starts[n - 1] : starts[n] - 1])
            and not self.source.in_region(starts[n - 1])
        ):
            n -= 1
        return max(0, n - 1), min(last, len(lines) - 1)

    def _emit(
        self, text: str, ends: bool = False, code: bool = False, inside: bool = False
    ) -> None:
        self.out.append(_Line(text, self.eol, self.keep, ends, code, inside))

    def _line(self, n: int) -> None:
        """Line `n`, formatted."""
        raw = self.lines[n]
        line = raw[:-1] if raw.endswith("\r") else raw
        self.eol = "\r\n" if len(line) < len(raw) else "\n"
        source, start = self.source, self.starts[n]
        end = start + len(line)
        if source.in_region(start):  # inside a tag begun on an earlier line: not ours to touch
            self._emit(line, source.ends_word(source.masked[start:end]), inside=True)
            return
        # Trailing whitespace, unless a tag carries on past this line.
        continues = source.in_region(end) or source.in_region(end - 1)
        body = line if continues else _TRAILING.sub("", line)
        masked = source.masked[start : start + len(body)]
        lead = _leading(body)
        if source.words_call(start + lead):
            self._emit(body, ends=True, code=True)  # whole words
            self.open = False
        elif not _strip(masked):
            self._emit(body)
        else:
            context = _Context(body, masked, lead, start)
            for piece in self._pieces(context):
                self._piece(piece, context)

    def _pieces(self, context: _Context) -> list[_Piece]:
        """A line's code split after each `;;` that has more code after it on the line."""
        body, masked = context.body, context.masked
        word_end = _lexicon().word_terminator
        pieces = []
        start = i = context.lead
        while i < len(masked):
            cut = i + len(word_end)
            if masked.startswith(word_end, i) and _strip(masked[cut:]):
                pieces.append(_Piece(start, body[start:cut]))
                start = cut + _leading(body[cut:])
                i = start - 1
            i += 1
        pieces.append(_Piece(start, body[start:]))
        return pieces

    def _piece(self, piece: _Piece, context: _Context) -> None:
        """One piece of a line: a label on a line of its own, then its code one indent in."""
        at, text = piece.at, piece.text
        masked = context.masked[at : at + len(text)]
        label = _lexicon().label.match(masked)
        if label is not None:
            size = label.end()
            rest = text[size:]
            # The label on its own line, with a trailing comment if that is all that
            # follows it; anything else after it is an ordinary piece.
            alone = not _strip(masked[size:])
            self._emit(text[:size] + (_TRAILING.sub("", rest) if alone else ""), code=True)
            if alone:
                self.open = False
                return
            skip = size + _leading(rest)
            at, text, masked = at + skip, text[skip:], masked[skip:]
        lead = context.lead
        deeper = self.open and at == lead and lead > len(self.indent)
        # A macro call that stands for whole words ends its word too, once a split
        # or a label has put it at the start of a line.
        words = at != lead and self.source.words_call(context.start + at)
        prefix = context.body[:lead] if deeper else self.indent
        self._emit(prefix + text, ends=self.source.ends_word(masked) or words, code=True)
        self.open = not _rstrip(masked).endswith(_lexicon().slot_separator)


def _format(text: str, indent: str, blank_line: bool, line_range: tuple[int, int] | None) -> str:
    """`text` formatted, or with only the lines of `line_range` formatted."""
    lines = text.split("\n")
    first, last = (0, len(lines) - 1) if line_range is None else line_range
    if first < 0 or last < first:
        raise ValueError(f"line range {line_range} is not first <= last, from 0")
    out = _Formatter(text, indent).run(first, last, blank_line)
    if line_range is None or first >= len(lines):
        return out if line_range is None else text
    last = min(last, len(lines) - 1)
    start = sum(len(line) + 1 for line in lines[:first])
    end = sum(len(line) + 1 for line in lines[:last]) + len(lines[last].removesuffix("\r"))
    return text[:start] + out + text[end:]


def format_source(
    text: str,
    *,
    indent: str = "    ",
    blank_line: bool = False,
    line_range: tuple[int, int] | None = None,
) -> str:
    """An assembly source in the repository's layout (see the module docstring).

    Args:
        text: The source, Jinja and all.
        indent: One level of indentation, for instruction lines.
        blank_line: Separate each `;;` word from the next by an empty line.
            Off by default: the kernels have none, and the editor draws a
            separator instead.
        line_range: (first, last), 0-based and inclusive: format only those
            lines, and keep the others as they are. None formats every line.

    Returns:
        The formatted source; `text` itself when there is nothing to change.

    Raises:
        FormatError: the formatted text is not the same program
            (`check_same_program`); the source is then not changed.
        ValueError: `line_range` is not first <= last.
    """
    out = _format(text, indent, blank_line, line_range)
    if out == text:
        return text
    check_same_program(text, out)
    return out


# --- The safety check ---

#: Grammar terminals after which whitespace changes nothing: no longer terminal
#: starts with them, so what follows lexes the same with or without a space.
_CLOSED = frozenset({"_SEMI2", "COLON"})


@functools.cache
def _scanner() -> re.Pattern[str]:
    """The grammar's terminals as one pattern, plus `JUNK` for any other character."""
    terminals = {t.name: t.pattern.to_regexp() for t in get_parser().terminals}
    order = (
        "COMMENT_HASH",
        "COMMENT_LINE",
        "_SEMI2",
        "_SEMI",
        "COLON",
        "TOKEN",
        "_NL",
        "WS_INLINE",
    )
    return re.compile(
        "|".join(f"(?P<{name}>{terminals[name]})" for name in order) + r"|(?P<JUNK>.)"
    )


def _stretches(text: str) -> Iterator[tuple[str, str]]:
    """(kind, text) of each Jinja tag (Jinja's own lexer) and each lexeme between them."""
    at = 0
    for span in tag_spans(text) if template.has_markers(text) else ():
        yield from _lexemes(text[at : span.start])
        yield "jinja", text[span.start : span.end]
        at = span.end
    yield from _lexemes(text[at:])


def _lexemes(text: str) -> Iterator[tuple[str, str]]:
    for match in _scanner().finditer(text):
        kind = match.lastgroup or "JUNK"
        value = match.group()
        yield (kind, value.rstrip(" \t\r") if kind in COMMENTS else value)


def _token_stream(text: str) -> list[tuple[str, str, bool]]:
    """The source as (kind, text, spaced): each Jinja tag, comment and token, and
    whether whitespace parts it from the item before, where that can matter.

    The space matters to the render (`lr{{ i }}` is not `lr {{ i }}`) except
    after a `;;` or a `:`; before the first item it never does. A comment is
    compared without its trailing whitespace, which the formatter removes.
    """
    stream: list[tuple[str, str, bool]] = []
    spaced = False
    for kind, value in _stretches(text):
        if kind in ("_NL", "WS_INLINE"):
            spaced = True
            continue
        matters = bool(stream) and stream[-1][0] not in _CLOSED
        stream.append((kind, value, spaced and matters))
        spaced = False
    return stream


def _words(text: str) -> tuple[int, ...] | None:
    """The words `text` assembles to, without its warnings; None if it does not assemble."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return AssemblySession(text).assemble().words
    except AssemblerError:
        return None


def check_same_program(before: str, after: str) -> None:
    """Raise `FormatError` unless `after` is `before` with only its layout changed.

    Both texts must read as the same stream of Jinja tags (from Jinja's own
    lexer), comments (up to trailing whitespace) and grammar tokens, with
    whitespace between two items wherever it was and the render could tell.
    A source without Jinja must also assemble to the same words, if it
    assembles at all: one being edited may not, and still gets formatted.

    Args:
        before: The source as it was.
        after: The source as formatted.

    Raises:
        FormatError: the two differ in more than layout.
    """
    old, new = _token_stream(before), _token_stream(after)
    if old != new:
        k = next((i for i, (a, b) in enumerate(zip(old, new, strict=False)) if a != b), None)
        if k is None:
            k = min(len(old), len(new))
        was = old[k][1] if k < len(old) else "the end"
        now = new[k][1] if k < len(new) else "the end"
        raise FormatError(
            f"formatting would change the program: item {k + 1} is {now!r}, not {was!r}"
        )
    if template.has_markers(before):
        return
    words = _words(before)
    if words is not None and _words(after) != words:
        raise FormatError("formatting would change the program: it assembles to other words")
