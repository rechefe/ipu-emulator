"""Every analysis the editor asks of one assembly source, answered by the assembler.

The editor needs four things for each version of a document: the diagnostics,
semantic tokens, whether each word's instructions fit its slots, and the Jinja
regions with the names they define and use. Each comes from the assembler's
own machinery, not from a copy of it:

- diagnostics from `ipu_as.diagnostics` (exactly what `check` reports);
- tokens from the grammar's lexer on the rendered text, mapped back to the
  source through the origin markers of `ipu_as.source_map`;
- word fit from `ipu_as.encode.assign_slots`, run on each word alone, so one
  word that does not fit hides no other;
- Jinja regions from Jinja's own lexer, and their names from Jinja's parser.

The source is rendered once, and that render serves every feature.

    result = analyse(text, want=frozenset({"diagnostics", "tokens"}))
    result.tokens[0].role

Positions are 0-based, matching LSP and the VS Code API. `ipu-as check --json
--json-version 2 --tokens --words --jinja` prints an `Analysis` (see
vscode-ipu-asm/schema/check-json.md).
"""

from __future__ import annotations

import bisect
import functools
import re
from collections import Counter
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from types import MappingProxyType

import lark

import ipu_as.template as template
from ipu_as.diagnostics import Diagnostic, check_rendered, render_for_diagnosis, tidy_message
from ipu_as.encode import WILDCARD_OPCODES, assign_slots
from ipu_as.errors import AssemblerError, SlotError
from ipu_as.ipu_token import NUMBER_PATTERN
from ipu_as.jinja_tags import TagSpan, header_names, is_params_header, tag_names, tag_spans
from ipu_as.source_map import source_line_at, strip_markers
from ipu_as.symbols import SymbolTable
from ipu_as.syntax import ASTBuilder, get_parser, parse_tree
from ipu_isa.codec.slot import OPCODE_SLOT
from ipu_isa.operand_types import OPERAND_TYPES, OperandKind
from ipu_isa.view import isa

__all__ = ["FEATURES", "Analysis", "JinjaRegion", "Range", "Token", "WordFit", "analyse"]

#: What `analyse` can be asked for, in the order an `Analysis` lists them.
FEATURES = ("diagnostics", "tokens", "words", "jinja")


@dataclass(frozen=True)
class Range:
    """A span of the source, 0-based; the end is exclusive."""

    line: int
    column: int
    end_line: int
    end_column: int


@dataclass(frozen=True)
class Token:
    """One semantic token, on one source line.

    Attributes:
        line: Its line.
        col: Its first column.
        length: Its length in characters.
        type: The grammar's terminal (`TOKEN`, `COMMENT_HASH`, `COMMENT_LINE`),
            or the Jinja region's kind (`comment`, `block`, `variable`).
        role: `mnemonic`, `register`, `enum`, `number`, `label-def`,
            `label-ref`, `comment` or `jinja`.
    """

    line: int
    col: int
    length: int
    type: str
    role: str


@dataclass(frozen=True)
class WordFit:
    """Whether one word's instructions fit its slots.

    Attributes:
        range: From the word's first token in the source to the end of its
            last; whole source lines when none of its tokens is in the source
            (all came from `{{ }}` expressions).
        mnemonics: The mnemonics as written, in order.
        slots: Slot name -> instructions in it (pseudo-instructions expanded).
            For a word that does not fit, the slots its instructions ask for.
        fits: True when every instruction gets a slot.
        error: Why it does not fit, as the assembler says it; None if it fits.
    """

    range: Range
    mnemonics: tuple[str, ...]
    slots: Mapping[str, int]
    fits: bool
    error: str | None


@dataclass(frozen=True)
class JinjaRegion:
    """One Jinja tag: a `{# #}` comment, a `{% %}` block or a `{{ }}` expression.

    Attributes:
        kind: `comment`, `block` or `variable`.
        range: Where it is; an unclosed tag runs to the end of the source.
        defines: The names it binds, in order: `set` targets, `for` targets,
            a macro and its parameters, `import` names; for the `ipu:params`
            header comment, the parameters it declares.
        uses: The names it reads, in order (Jinja globals such as `range`
            and `loop` included).
        start: Its first character, as an offset in the source.
        end: The offset just past it.
    """

    kind: str
    range: Range
    defines: tuple[str, ...]
    uses: tuple[str, ...]
    start: int
    end: int


@dataclass(frozen=True)
class Analysis:
    """What `analyse` found; a feature not asked for is None."""

    diagnostics: tuple[Diagnostic, ...] | None = None
    tokens: tuple[Token, ...] | None = None
    words: tuple[WordFit, ...] | None = None
    jinja: tuple[JinjaRegion, ...] | None = None

    def to_dict(self) -> dict[str, list[dict]]:
        """The features asked for, as JSON-ready lists, keyed by feature name."""
        return {
            name: [_jsonable(item) for item in getattr(self, name)]
            for name in FEATURES
            if getattr(self, name) is not None
        }


def _jsonable(item: object) -> dict:
    if isinstance(item, Diagnostic):
        return item.to_dict()
    if isinstance(item, WordFit):
        return {
            "range": asdict(item.range),
            "mnemonics": list(item.mnemonics),
            "slots": dict(item.slots),
            "fits": item.fits,
            "error": item.error,
        }
    return asdict(item)  # type: ignore[call-overload]


class _Document:
    """One source, and what analysing it needs, each computed at most once."""

    def __init__(self, text: str, inputs: template.TemplateInputs, target: str) -> None:
        self.text = text
        self.inputs = inputs
        self.target = target
        #: Without Jinja the source is its own render, and positions carry over.
        self.literal = not template.has_markers(text)

    @cached_property
    def line_starts(self) -> list[int]:
        return [0] + [m.end() for m in re.finditer("\n", self.text)]

    def line_end(self, line: int) -> int:
        """The offset of the end of `line`, before its newline."""
        starts = self.line_starts
        return starts[line + 1] - 1 if line + 1 < len(starts) else len(self.text)

    def position(self, offset: int) -> tuple[int, int]:
        line = bisect.bisect_right(self.line_starts, offset) - 1
        return line, offset - self.line_starts[line]

    def range(self, start: int, end: int) -> Range:
        return Range(*self.position(start), *self.position(end))

    def pieces(self, start: int, end: int) -> Iterator[tuple[int, int, int]]:
        """[start, end) split at line ends, as non-empty (line, column, length)."""
        while start < end:
            line, column = self.position(start)
            stop = min(end, self.line_end(line))
            if stop > start:
                yield line, column, stop - start
            start = stop + 1

    @cached_property
    def spans(self) -> tuple[TagSpan, ...]:
        return () if self.literal else tag_spans(self.text)

    @cached_property
    def span_ends(self) -> list[int]:
        return [span.end for span in self.spans]

    def overlaps(self, start: int, end: int) -> bool:
        """True when [start, end) touches a Jinja region."""
        i = bisect.bisect_right(self.span_ends, start)
        return i < len(self.spans) and self.spans[i].start < end

    def outside_spans(self, start: int, end: int) -> Iterator[tuple[int, int]]:
        """The parts of [start, end) that are not in a Jinja region."""
        for span in self.spans[bisect.bisect_right(self.span_ends, start) :]:
            if span.start >= end:
                break
            if span.start > start:
                yield start, span.start
            start = max(start, span.end)
        if start < end:
            yield start, end

    @cached_property
    def span_starts(self) -> list[int]:
        return [span.start for span in self.spans]

    def next_span_start(self, offset: int) -> int:
        i = bisect.bisect_left(self.span_starts, offset)
        return self.spans[i].start if i < len(self.spans) else len(self.text)

    @cached_property
    def rendered(self) -> tuple[str, Diagnostic | None]:
        """The render `ipu_as.diagnostics` checks, and its template diagnostic if it failed."""
        return render_for_diagnosis(self.text, self.inputs)

    @cached_property
    def placed(self) -> tuple[list[lark.Token], dict[int, list[tuple[int, int, int]]]] | None:
        """The render's tokens and their places in the source (`_placements`); None if it failed."""
        rendered, failed = self.rendered
        return None if failed is not None else _placements(self, rendered)

    def source_line(self, rendered: str, token: lark.Token) -> int | None:
        """The source line a rendered token came from, if the render says."""
        if self.literal:
            return token.line - 1 if token.line is not None else None
        line = source_line_at(rendered, token.start_pos or 0)
        return line if line is not None and line < len(self.line_starts) else None


# --- Diagnostics ---


def _diagnostics(document: _Document) -> tuple[Diagnostic, ...]:
    rendered, failed = document.rendered
    if failed is not None:
        return (failed,)
    return tuple(check_rendered(document.text, rendered, document.inputs, document.target))


# --- Tokens ---

_SEPARATORS = ("_SEMI", "_SEMI2")
_COMMENTS = ("COMMENT_HASH", "COMMENT_LINE")
_NUMBER = re.compile(NUMBER_PATTERN)
_REGISTERS = frozenset(
    member.lower()
    for operand_type in OPERAND_TYPES.values()
    if operand_type.kind is OperandKind.REGISTER
    for member in operand_type.domain.members
)
_ROLE_OF_KIND = {
    OperandKind.REGISTER: "register",
    OperandKind.ENUM: "enum",
    OperandKind.IMMEDIATE: "number",
    OperandKind.LABEL: "label-ref",
}


@functools.cache
def _operand_kinds(mnemonic: str, arity: int) -> tuple[OperandKind, ...]:
    """The kind of each operand of `mnemonic` with `arity` operands; () if it is unknown."""
    pseudo = isa().pseudo(mnemonic, arity)
    if pseudo is not None:
        return tuple(operand.type.kind for operand in pseudo.operands)
    try:
        slot = isa().find_slot(mnemonic)
    except (KeyError, ValueError):
        return ()
    return tuple(operand.type.kind for operand in isa().instruction(slot, mnemonic).operands)


def _operand_role(kinds: tuple[OperandKind, ...], index: int, text: str) -> str:
    """An operand's role: a number literal is a number; else what its operand type says."""
    if _NUMBER.fullmatch(text):
        return "number"
    if index < len(kinds):
        return _ROLE_OF_KIND[kinds[index]]
    return "register" if text.lower() in _REGISTERS else "label-ref"


def _roles(tokens: Sequence[lark.Token]) -> list[str | None]:
    """Each token's role, from its place in the token stream (None for a separator).

    The grammar is `[label:] mnemonic operand* (; ...)* ;;`, with newlines
    insignificant, so the place of a token in the stream is its place in the
    parse tree. Unlike the tree, the stream exists for a text that does not parse.
    """
    roles: list[str | None] = [None] * len(tokens)
    code = [i for i, token in enumerate(tokens) if token.type not in _COMMENTS]
    instruction: list[int] = []

    def close() -> None:
        if instruction:
            mnemonic, *operands = instruction
            roles[mnemonic] = "mnemonic"
            kinds = _operand_kinds(str(tokens[mnemonic]), len(operands))
            for index, at in enumerate(operands):
                roles[at] = _operand_role(kinds, index, str(tokens[at]))
            instruction.clear()

    for k, i in enumerate(code):
        kind = tokens[i].type
        following = tokens[code[k + 1]].type if k + 1 < len(code) else None
        if kind == "TOKEN" and following == "COLON":
            close()
            roles[i] = "label-def"
        elif kind == "TOKEN":
            instruction.append(i)
        elif kind in _SEPARATORS:
            close()
    close()
    for i, token in enumerate(tokens):
        if token.type in _COMMENTS:
            roles[i] = "comment"
    return roles


def _lexed(rendered: str) -> list[lark.Token]:
    """The grammar's tokens of `rendered`, without whitespace, up to any it cannot lex."""
    tokens: list[lark.Token] = []
    try:
        for token in get_parser().lex(rendered, dont_ignore=True):
            if token.type not in ("WS_INLINE", "_NL"):
                tokens.append(token)
    except lark.exceptions.LarkError:
        pass  # a character the grammar has no token for: the tokens before it stand
    return tokens


#: A character that continues a TOKEN.
_WORD_CHAR = re.compile(r"[A-Za-z0-9_.]")


def _joins(left: str, right: str) -> bool:
    """True when two adjacent characters would lex as one token."""
    return bool(_WORD_CHAR.match(left) and _WORD_CHAR.match(right)) or left == right == ";"


class _Locator:
    """Places rendered tokens in the source.

    A token's line comes from the origin markers; its column from finding its
    text on that line, after the last token placed there from the same rendered
    line. A token that is not literally on its line came from a `{{ }}`
    expression (or an include), and is not placed: its Jinja region covers it.
    """

    def __init__(self, document: _Document, rendered: str) -> None:
        self.document = document
        self.rendered = rendered
        self.cursors: dict[int, int] = {}
        self.rendered_line: int | None = None

    def place(self, token: lark.Token) -> list[tuple[int, int, int]]:
        """Where `token` is in the source, as (line, column, length) pieces."""
        if token.line != self.rendered_line:
            # A new rendered line, perhaps the next pass of a loop over the same source line.
            self.cursors.clear()
            self.rendered_line = token.line
        line = self.document.source_line(self.rendered, token)
        if line is None:
            return []
        if token.type in _COMMENTS:
            return self._comment(line, strip_markers(str(token)).rstrip())
        return self._code(line, str(token))

    def _code(self, line: int, value: str) -> list[tuple[int, int, int]]:
        text, first = self.document.text, self.document.line_starts[line]
        end = self.document.line_end(line)
        at = text.find(value, self.cursors.get(line, first), end)
        while at >= 0 and not self._alone(at, at + len(value)):
            at = text.find(value, at + 1, end)
        if at < 0:
            return []
        self.cursors[line] = at + len(value)
        return [(line, at - first, len(value))]

    def _alone(self, start: int, end: int) -> bool:
        """True when [start, end) is a whole token of the source, outside Jinja."""
        text = self.document.text
        before = text[start - 1] if start > 0 else " "
        after = text[end] if end < len(text) else " "
        if _joins(before, text[start]) or _joins(text[end - 1], after):
            return False
        return not self.document.overlaps(start, end)

    def _comment(self, line: int, value: str) -> list[tuple[int, int, int]]:
        """A comment runs to the end of its source line, around the Jinja inside it."""
        if not value:
            return []  # only an origin marker
        document, text = self.document, self.document.text
        end = document.line_end(line)
        opener = "//" if value.startswith("//") else "#"
        at = text.find(opener, self.cursors.get(line, document.line_starts[line]), end)
        while at >= 0:
            fragment = text[at : min(document.next_span_start(at), end)].rstrip()
            if not document.overlaps(at, at + 1) and value.startswith(fragment):
                self.cursors[line] = end
                stop = at + len(text[at:end].rstrip())
                return [
                    piece
                    for start, finish in document.outside_spans(at, stop)
                    for piece in document.pieces(start, finish)
                ]
            at = text.find(opener, at + 1, end)
        return []


def _placements(document: _Document, rendered: str) -> tuple[list[lark.Token], dict]:
    """The rendered text's tokens, and where each is in the source, by rendered offset."""
    lexed = _lexed(rendered)
    locator = _Locator(document, rendered)
    return lexed, {token.start_pos: locator.place(token) for token in lexed}


def _tokens(document: _Document) -> tuple[Token, ...]:
    """The semantic tokens: the placed assembly tokens and the Jinja regions, line by line."""
    found: dict[tuple[int, int, int], Token] = {}
    if document.placed is not None:
        lexed, placed = document.placed
        for token, role in zip(lexed, _roles(lexed), strict=True):
            for line, col, length in placed[token.start_pos] if role is not None else ():
                # A loop renders a source line many times: keep the first.
                found.setdefault((line, col, length), Token(line, col, length, token.type, role))
    for span in document.spans:
        for line, col, length in document.pieces(span.start, span.end):
            found.setdefault((line, col, length), Token(line, col, length, span.kind, "jinja"))
    return tuple(sorted(found.values(), key=lambda token: (token.line, token.col)))


# --- Words ---


def _word_range(document: _Document, rendered: str, tree: lark.Tree) -> Range | None:
    """From the word's first placed token to the end of its last one.

    A word none of whose tokens is in the source (all from `{{ }}`) spans the
    whole source lines its first and last tokens came from.
    """
    tokens = list(tree.scan_values(lambda value: isinstance(value, lark.Token)))
    _, placed = document.placed or ((), {})
    pieces = [piece for token in tokens for piece in placed.get(token.start_pos, ())]
    if pieces:
        (line, col, _), (end_line, end_col, length) = pieces[0], pieces[-1]
        return Range(line, col, end_line, end_col + length)
    first = document.source_line(rendered, tokens[0])
    last = document.source_line(rendered, tokens[-1])
    if first is None or last is None:
        return None
    last = max(first, last)
    return Range(first, 0, last, document.line_end(last) - document.line_starts[last])


def _slots_asked(compound: dict) -> Mapping[str, int]:
    """Slot -> instructions that name it (a `NOP` takes any slot, so it names none)."""
    names = (instruction["opcode"].token.value for instruction in compound["instructions"])
    return MappingProxyType(
        Counter(
            str(OPCODE_SLOT[name.upper()])
            for name in names
            if name.lower() not in WILDCARD_OPCODES and name.upper() in OPCODE_SLOT
        )
    )


def _fit(word: Range, tree: lark.Tree) -> WordFit:
    """Slot assignment of one compound instruction, built on its own."""
    mnemonics = tuple(str(instruction.children[0]) for instruction in tree.find_data("instruction"))
    try:
        # Its own symbol table: a label defined twice is no slot problem.
        compound = ASTBuilder(SymbolTable()).transform(tree)
    except lark.exceptions.VisitError as error:
        if not isinstance(error.orig_exc, AssemblerError):
            raise error.orig_exc from error
        return WordFit(word, mnemonics, MappingProxyType({}), False, _message(error.orig_exc))
    try:
        placed = assign_slots(compound)
    except SlotError as error:
        return WordFit(word, mnemonics, _slots_asked(compound), False, _message(error))
    slots = MappingProxyType(Counter(str(placement.slot) for placement in placed))
    return WordFit(word, mnemonics, slots, True, None)


def _message(error: AssemblerError) -> str:
    return tidy_message(str(error))


def _words(document: _Document) -> tuple[WordFit, ...]:
    """The slot fit of each word the source renders to, each placed on its source lines."""
    rendered, failed = document.rendered
    if failed is not None:
        return ()
    try:
        tree = parse_tree(rendered)
    except lark.exceptions.LarkError:
        return ()  # the parse diagnostic says why
    found: dict[tuple, WordFit] = {}
    for compound in tree.iter_subtrees_topdown():
        if compound.data != "compound_instr_list":
            continue
        word = _word_range(document, rendered, compound)
        # A loop renders a word many times; the same text in the same place fits the same.
        key = (word, tuple(map(str, compound.scan_values(lambda v: isinstance(v, lark.Token)))))
        if word is not None and key not in found:
            found[key] = _fit(word, compound)
    return tuple(found.values())


# --- Jinja ---


def _jinja(document: _Document) -> tuple[JinjaRegion, ...]:
    """Each Jinja region, with the names it defines and uses."""
    header = header_names(document.text)
    regions = []
    for span in document.spans:
        tag = document.text[span.start : span.end]
        if span.kind != "comment":
            defines, uses = tag_names(tag)
        elif header and is_params_header(tag):
            defines, uses, header = header, (), ()
        else:
            defines, uses = (), ()
        where = document.range(span.start, span.end)
        regions.append(JinjaRegion(span.kind, where, defines, uses, span.start, span.end))
    return tuple(regions)


_ANALYSES: Mapping[str, Callable[[_Document], tuple]] = {
    "diagnostics": _diagnostics,
    "tokens": _tokens,
    "words": _words,
    "jinja": _jinja,
}


def analyse(
    text: str,
    *,
    defines: Mapping[str, object] | None = None,
    search_path: Sequence[Path | str] = (),
    target: str = "emulator",
    want: Collection[str] = frozenset({"diagnostics"}),
) -> Analysis:
    """Analyse one assembly source: render it once and answer each feature in `want`.

    Args:
        text: The source, as the editor shows it.
        defines: Template values, as for the assembler.
        search_path: Roots `{% include %}` and `{% import %}` resolve in.
        target: `"emulator"` or `"hardware"`, for the diagnostics.
        want: The features to compute, from `FEATURES`: `diagnostics` (the
            same as `ipu_as.diagnostics.check`), `tokens`, `words`, `jinja`.

    Returns:
        The analysis; each feature not in `want` is None.

    Raises:
        ValueError: `want` names a feature that does not exist.
    """
    unknown = sorted(set(want) - set(FEATURES))
    if unknown:
        raise ValueError(f"unknown analysis feature(s) {unknown}; have {list(FEATURES)}")
    inputs = template.TemplateInputs(
        defines=dict(defines or {}), search_path=tuple(Path(p) for p in search_path)
    )
    document = _Document(text, inputs, target)
    return Analysis(**{name: _ANALYSES[name](document) for name in FEATURES if name in want})
