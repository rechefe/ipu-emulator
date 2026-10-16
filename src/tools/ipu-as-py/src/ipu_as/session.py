"""One assembly, from source text to instruction words.

`AssemblySession` runs the pipeline: render the Jinja layer (once), parse,
resolve labels, encode. It owns its `SymbolTable`, so assemblies in one process
never interfere. Every failure is an `AssemblerError` with a location:
`TemplateError` from rendering, `ParseError` from the grammar, and an
`EncodeError` subclass from encoding (a duplicate label is found while parsing,
but is a `LabelError` all the same).

    result = AssemblySession(source).assemble()
    result.words, result.to_bin(), result.symbols
"""

from __future__ import annotations

import collections
import hashlib
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import lark

import ipu_as.compound_inst as compound_inst
import ipu_as.template as template
from ipu_as.errors import AssemblerError, ParseError, SourceLocation, TemplateError
from ipu_as.symbols import SymbolTable
from ipu_as.syntax import ASTBuilder, parse_tree

__all__ = [
    "AssemblyResult",
    "AssemblySession",
    "SymbolTable",
    "assemble",
    "instruction_aligned_bytes_len",
]

_WORD_SIZE_BITS = 32


def instruction_aligned_bytes_len() -> int:
    """Bytes per instruction word in a `.bin` file: the word rounded up to 32 bits."""
    bits = compound_inst.CompoundInst.bits()
    if bits % _WORD_SIZE_BITS != 0:
        bits += _WORD_SIZE_BITS - (bits % _WORD_SIZE_BITS)
    return (bits // _WORD_SIZE_BITS) * 4


@dataclass(frozen=True)
class AssemblyResult:
    """The output of one assembly."""

    words: tuple[int, ...]
    symbols: Mapping[str, int]
    rendered: str
    #: Each warning the encoder raised, as text (simulation-only slots).
    warnings: tuple[str, ...]
    _warning_categories: tuple[type[Warning], ...] = field(default=(), repr=False, compare=False)

    def to_mem(self) -> str:
        """The `.mem` file text: one `0x0<8 hex digits>` line per word."""
        return "".join(f"0x0{word:08x}\n" for word in self.words)

    def to_bin(self) -> bytes:
        """The `.bin` file bytes: each word little-endian, padded to 32 bits."""
        width = instruction_aligned_bytes_len()
        return b"".join(word.to_bytes(width, byteorder="little") for word in self.words)

    def reissue_warnings(self) -> None:
        """Raise the assembly's warnings again, as encoding did, for a caller handed this result."""
        for message, category in zip(self.warnings, self._warning_categories, strict=True):
            warnings.warn(message, category, stacklevel=3)


class AssemblySession:
    """Render, parse and encode one assembly source."""

    def __init__(
        self,
        source: str,
        *,
        defines: Mapping[str, object] | None = None,
        name: str = "<input>",
    ) -> None:
        self.source = source
        self.defines = dict(defines or {})
        self.name = name
        self.symbols = SymbolTable()
        self._rendered: str | None = None
        self._parsed: list[dict] | None = None

    def render(self) -> str:
        """The source with its Jinja layer rendered; rendered at most once. Raises TemplateError."""
        if self._rendered is None:
            if self.defines:
                raise TemplateError("defines not supported yet")
            if not template.has_markers(self.source):
                self._rendered = self.source
            else:
                try:
                    self._rendered = template.render(self.source)
                except Exception as error:
                    raise TemplateError(str(error)) from error
        return self._rendered

    def parse(self) -> list[dict]:
        """The compound instructions, with every label defined. Raises ParseError or LabelError."""
        if self._parsed is None:
            rendered = self.render()
            try:
                self._parsed = ASTBuilder(self.symbols).transform(parse_tree(rendered))
            except lark.exceptions.VisitError as error:
                # lark wraps whatever the AST builder raises; a LabelError or
                # OperandError found while building is reported as itself.
                if isinstance(error.orig_exc, AssemblerError):
                    raise error.orig_exc from error
                raise ParseError(str(error)) from error
            except lark.exceptions.LarkError as error:
                raise ParseError(str(error), location=_lark_location(error)) from error
        return self._parsed

    def assemble(self) -> AssemblyResult:
        """Encode every instruction word. Raises TemplateError, ParseError or an EncodeError."""
        parsed = self.parse()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            words = tuple(compound_inst.CompoundInst(instr).encode() for instr in parsed)
        result = AssemblyResult(
            words=words,
            symbols=MappingProxyType(dict(self.symbols.items())),
            rendered=self.render(),
            warnings=tuple(str(w.message) for w in caught),
            _warning_categories=tuple(w.category for w in caught),
        )
        result.reissue_warnings()
        return result


def _lark_location(error: lark.exceptions.LarkError) -> SourceLocation | None:
    line = getattr(error, "line", None)
    column = getattr(error, "column", None)
    if isinstance(line, int) and isinstance(column, int) and line > 0:
        return SourceLocation(line, column)
    return None


#: Results of recent assemblies, so a test suite that assembles the same kernel
#: many times in one process encodes it once.
_CACHE_SIZE = 128
_cache: collections.OrderedDict[tuple[str, tuple], AssemblyResult] = collections.OrderedDict()


def _cache_key(source: str, defines: Mapping[str, object] | None) -> tuple[str, tuple]:
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    return digest, tuple(sorted((defines or {}).items()))


def assemble(
    source: str, *, defines: Mapping[str, object] | None = None, name: str = "<input>"
) -> AssemblyResult:
    """Assemble `source`, reusing the result of an identical recent assembly.

    A cached result raises its warnings again, so a caller sees the same
    warnings whether or not the result came from the cache.
    """
    key = _cache_key(source, defines)
    cached = _cache.get(key)
    if cached is not None:
        _cache.move_to_end(key)
        cached.reissue_warnings()
        return cached
    result = AssemblySession(source, defines=defines, name=name).assemble()
    _cache[key] = result
    if len(_cache) > _CACHE_SIZE:
        _cache.popitem(last=False)
    return result
