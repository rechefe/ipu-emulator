"""Every analysis the editor asks of one assembly source, answered by the assembler.

The editor needs four things for each version of a document: the diagnostics,
semantic tokens, whether each word's instructions fit its slots, and the Jinja
regions with the names they define and use. Each comes from the assembler's
own machinery, not from a copy of it:

- diagnostics from `ipu_as.diagnostics` (exactly what `check` reports);
- tokens from the grammar's lexer on the rendered text, placed in the source
  by `ipu_as.placement` (through the origin markers of `ipu_as.source_map`);
- word fit from `ipu_as.encode.assign_slots`, run on each word alone, so one
  word that does not fit hides no other;
- Jinja regions from Jinja's own lexer, and their names from Jinja's parser.

The source is rendered once, and that render serves every feature.

    result = analyse(text, want=frozenset({"diagnostics", "tokens"}))
    result.tokens[0].role

Positions are 0-based lines and columns, as in LSP and the VS Code API, but a
column counts code points (Python string indices), as the version 1
diagnostics do, not the UTF-16 code units those APIs count: the editor
converts. `ipu-as check --json --json-version 2 --tokens --words --jinja`
prints an `Analysis` (see vscode-ipu-asm/schema/check-json.md).
"""

from __future__ import annotations

import functools
import re
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
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
from ipu_as.jinja_tags import header_names, is_params_header, tag_names
from ipu_as.placement import COMMENTS, Piece, Rendering, Source
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
    """A span of the source, 0-based; the end is exclusive.

    Attributes:
        line: The line it starts on.
        column: Its first column, in code points.
        end_line: The line it ends on.
        end_column: The column just past its last character.
    """

    line: int
    column: int
    end_line: int
    end_column: int


@dataclass(frozen=True)
class Token:
    """One semantic token, on one source line.

    Attributes:
        line: Its line.
        column: Its first column, in code points.
        length: Its length in code points; never past its line's end.
        type: The grammar's terminal (`TOKEN`, `COMMENT_HASH`, `COMMENT_LINE`),
            or the Jinja region's kind (`comment`, `block`, `variable`).
        role: `mnemonic`, `register`, `enum`, `number`, `label-def`,
            `label-ref`, `comment` or `jinja`.
    """

    line: int
    column: int
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
        slots: Slot -> instructions in it (pseudo-instructions expanded).
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
    """What `analyse` found; a feature not asked for is None.

    Attributes:
        diagnostics: As `ipu_as.diagnostics.check` reports them.
        tokens: The semantic tokens, sorted by line and column.
        words: Each instruction word's slot fit, in source order.
        jinja: Each Jinja region, in source order.
    """

    diagnostics: tuple[Diagnostic, ...] | None = None
    tokens: tuple[Token, ...] | None = None
    words: tuple[WordFit, ...] | None = None
    jinja: tuple[JinjaRegion, ...] | None = None

    def to_dict(self) -> dict[str, list[dict]]:
        """The features asked for, as JSON (`check --json --json-version 2`).

        Returns:
            Feature name -> its entries as JSON-ready dicts, for each feature
            asked for, in `FEATURES` order.
        """
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
        self.source = Source(text)

    def range(self, start: int, end: int) -> Range:
        return Range(*self.source.position(start), *self.source.position(end))

    @cached_property
    def rendered(self) -> tuple[str, Diagnostic | None]:
        """The render `ipu_as.diagnostics` checks, and its template diagnostic if it failed."""
        return render_for_diagnosis(self.text, self.inputs)

    @cached_property
    def rendering(self) -> Rendering:
        return Rendering(self.source, self.rendered[0])

    @cached_property
    def placed(self) -> tuple[list[lark.Token], dict[int, list[Piece]]] | None:
        """The render's tokens and their places in the source; None if it failed."""
        if self.rendered[1] is not None:
            return None
        lexed = _lexed(self.rendered[0])
        return lexed, self.rendering.place(lexed)


# --- Diagnostics ---


def _diagnostics(document: _Document) -> tuple[Diagnostic, ...]:
    rendered, failed = document.rendered
    if failed is not None:
        return (failed,)
    return tuple(check_rendered(document.text, rendered, document.inputs, document.target))


# --- Tokens ---

_SEPARATORS = ("_SEMI", "_SEMI2")
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
    code = [i for i, token in enumerate(tokens) if token.type not in COMMENTS]
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
        if token.type in COMMENTS:
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


def _tokens(document: _Document) -> tuple[Token, ...]:
    """The semantic tokens: the placed assembly tokens and the Jinja regions, line by line."""
    found: dict[tuple[int, int, int], Token] = {}
    if document.placed is not None:
        lexed, placed = document.placed
        for token, role in zip(lexed, _roles(lexed), strict=True):
            for line, column, length in placed[token.start_pos] if role is not None else ():
                # A loop renders a source line many times: keep the first.
                found.setdefault(
                    (line, column, length), Token(line, column, length, token.type, role)
                )
    for span in document.source.spans:
        for line, column, length in document.source.pieces(span.start, span.end):
            found.setdefault(
                (line, column, length), Token(line, column, length, span.kind, "jinja")
            )
    return tuple(sorted(found.values(), key=lambda token: (token.line, token.column)))


# --- Words ---


def _word_range(document: _Document, tokens: list[lark.Token]) -> tuple[Range, bool] | None:
    """(Where the word is, whether a token of it is in the source); None if unknown.

    The range runs from the word's first placed token to the end of its last
    one. A word none of whose tokens is in the source (all from `{{ }}`) spans
    the whole source lines its first and last tokens came from.
    """
    _, placed = document.placed or ((), {})
    pieces = [piece for token in tokens for piece in placed.get(token.start_pos, ())]
    if pieces:
        (line, column, _), (end_line, end_column, length) = pieces[0], pieces[-1]
        return Range(line, column, end_line, end_column + length), True
    first = document.rendering.source_line(tokens[0])
    last = document.rendering.source_line(tokens[-1])
    if first is None or last is None:
        return None
    last = max(first, last)
    source = document.source
    return Range(first, 0, last, source.line_end(last) - source.line_starts[last]), False


def _slots_asked(built: dict) -> Mapping[str, int]:
    """Slot -> instructions that name it (a `NOP` takes any slot, so it names none)."""
    names = (instruction["opcode"].token.value for instruction in built["instructions"])
    return MappingProxyType(
        Counter(
            str(OPCODE_SLOT[name.upper()])
            for name in names
            if name.lower() not in WILDCARD_OPCODES and name.upper() in OPCODE_SLOT
        )
    )


def _fit(word: Range, tree: lark.Tree) -> WordFit:
    """Slot assignment of one instruction word, built on its own."""
    mnemonics = tuple(str(instruction.children[0]) for instruction in tree.find_data("instruction"))
    try:
        # Its own symbol table: a label defined twice is no slot problem.
        built = ASTBuilder(SymbolTable()).transform(tree)
    except lark.exceptions.VisitError as error:
        if not isinstance(error.orig_exc, AssemblerError):
            raise error.orig_exc from error
        return WordFit(word, mnemonics, MappingProxyType({}), False, _message(error.orig_exc))
    try:
        placed = assign_slots(built)
    except SlotError as error:
        return WordFit(word, mnemonics, _slots_asked(built), False, _message(error))
    slots = MappingProxyType(Counter(str(placement.slot) for placement in placed))
    return WordFit(word, mnemonics, slots, True, None)


def _message(error: AssemblerError) -> str:
    return tidy_message(str(error))


def _words(document: _Document) -> tuple[WordFit, ...]:
    """The slot fit of each word in the source, once however often loops render it.

    Two renderings with the same range are one word of the source, rendered
    twice: a placed token is at one place in the source. Words placed on whole
    lines (no token in the source) are told apart by their order on their
    rendered line, so `{{ 'HALT;; HALT;;' }}` is two words, and a loop's next
    pass, on a rendered line of its own, renders the same ones again. Of the
    renderings of one word, the first that does not fit is kept, else the first.
    """
    rendered, failed = document.rendered
    if failed is not None:
        return ()
    try:
        tree = parse_tree(rendered)
    except lark.exceptions.LarkError:
        return ()  # the parse diagnostic says why
    found: dict[tuple, WordFit] = {}
    on_line: Counter[tuple] = Counter()
    for subtree in tree.iter_subtrees_topdown():
        if subtree.data != "compound_instr_list":
            continue
        tokens = list(subtree.scan_values(lambda value: isinstance(value, lark.Token)))
        where = _word_range(document, tokens)
        if where is None:
            continue
        word, in_source = where
        key: tuple = (word,)
        if not in_source:
            key = (word, on_line[word, tokens[-1].line])
            on_line[word, tokens[-1].line] += 1
        if key not in found or found[key].fits:
            fit = _fit(word, subtree)
            if key not in found or not fit.fits:
                found[key] = fit
    return tuple(found.values())


# --- Jinja ---


def _jinja(document: _Document) -> tuple[JinjaRegion, ...]:
    """Each Jinja region, with the names it defines and uses."""
    header = header_names(document.text)
    regions = []
    for span in document.source.spans:
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
