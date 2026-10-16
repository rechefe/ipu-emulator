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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import lark

import ipu_as.compound_inst as compound_inst
import ipu_as.template as template
from ipu_as.errors import AssemblerError, ParseError, SourceLocation
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
        """Raise the assembly's warnings again, as encoding did.

        Called by the public `assemble` functions, so each warning is attributed
        to their caller, whether the result is new or came from the cache.
        """
        for message, category in zip(self.warnings, self._warning_categories, strict=True):
            warnings.warn(message, category, stacklevel=3)


class AssemblySession:
    """Render, parse and encode one assembly source."""

    def __init__(
        self,
        source: str,
        *,
        defines: Mapping[str, object] | None = None,
        params: template.ParamSpec | None = None,
        search_path: Sequence[Path] = (),
        abi: Mapping[str, Mapping[str, str]] | None = None,
        name: str = "<input>",
    ) -> None:
        self.source = source
        self.abi = {namespace: dict(names) for namespace, names in (abi or {}).items()}
        self.defines = dict(defines or {})
        self.params = params
        self.search_path = tuple(Path(p) for p in search_path)
        self.name = name
        self.symbols = SymbolTable()
        self._rendered: str | None = None
        self._parsed: list[dict] | None = None

    def render(self) -> str:
        """The source with its Jinja layer rendered, at most once (`ipu_as.template.render`).

        Raises:
            TemplateError: a bad define or a jinja failure, located.
        """
        if self._rendered is None:
            inputs = template.TemplateInputs(
                defines=self.defines,
                params=self.params,
                search_path=self.search_path,
                namespaces=self.abi,
            )
            self._rendered = template.render(self.source, inputs, source_name=self.name)
        return self._rendered

    def parse(self) -> list[dict]:
        """The compound instructions of the rendered source, parsed once.

        Raises:
            TemplateError, ParseError, or the LabelError / OperandError found
            while building the instruction list.
        """
        if self._parsed is None:
            self._parsed = self.parse_rendered(self.render())
        return self._parsed

    def parse_rendered(self, rendered: str) -> list[dict]:
        """Parse text that is already rendered, into a fresh symbol table.

        The stage `parse` runs on this session's own render; diagnostics passes
        an instrumented render instead. It discards any parse `parse` cached,
        since that parse's labels are no longer in `symbols`.

        Raises:
            ParseError: the text does not match the grammar.
            LabelError: a label is defined twice.
            OperandError: an operand is invalid while building the instruction list.
        """
        self.symbols = SymbolTable()
        self._parsed = None  # the new symbol table belongs to this parse, not a cached one
        try:
            return ASTBuilder(self.symbols).transform(parse_tree(rendered))
        except lark.exceptions.VisitError as error:
            # lark wraps whatever the AST builder raises: an assembler error is
            # reported as itself, anything else is a bug and raised as itself.
            raise error.orig_exc from error
        except lark.exceptions.LarkError as error:
            raise ParseError(str(error), location=_lark_location(error)) from error

    def encode(self, parsed: list[dict]) -> tuple[int, ...]:
        """The instruction words of `parsed` (from `parse` or `parse_rendered`).

        Raises:
            EncodeError: an operand, slot or label error.
        """
        return tuple(compound_inst.CompoundInst(instr).encode() for instr in parsed)

    def assemble(self) -> AssemblyResult:
        """Encode every instruction word and raise the encoder's warnings.

        Raises:
            TemplateError, ParseError or an EncodeError subclass.
        """
        result = self._encode()
        result.reissue_warnings()
        return result

    def _encode(self) -> AssemblyResult:
        """Encode without raising warnings; they are recorded in the result."""
        parsed = self.parse()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                words = self.encode(parsed)
                failure = None
            except AssemblerError as error:
                failure = error
        if failure is not None:
            # Raise what encoding warned about before it failed, then the error.
            for warning in caught:
                warnings.warn(warning.message, warning.category, stacklevel=3)
            raise failure
        result = AssemblyResult(
            words=words,
            symbols=MappingProxyType(dict(self.symbols.items())),
            rendered=self.render(),
            warnings=tuple(str(w.message) for w in caught),
            _warning_categories=tuple(w.category for w in caught),
        )
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
_cache: collections.OrderedDict[tuple, AssemblyResult] = collections.OrderedDict()


def _cache_key(
    source: str,
    defines: Mapping[str, object] | None,
    params: template.ParamSpec | None,
    abi: Mapping[str, Mapping[str, str]] | None,
) -> tuple | None:
    """The cache key, or None when the inputs cannot be one (a ParamSpec with an unhashable default)."""
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    # Each value with its type and repr: True == 1 == 1.0 in Python, but they
    # bind differently against a declared int (only 1 is valid).
    values = tuple(sorted((k, type(v).__name__, repr(v)) for k, v in (defines or {}).items()))
    names = tuple(
        sorted((namespace, tuple(sorted(regs.items()))) for namespace, regs in (abi or {}).items())
    )
    key = (digest, values, params, names)
    try:
        hash(key)
    except TypeError:
        return None
    return key


def assemble(
    source: str,
    *,
    defines: Mapping[str, object] | None = None,
    params: template.ParamSpec | None = None,
    search_path: Sequence[Path] = (),
    abi: Mapping[str, Mapping[str, str]] | None = None,
    name: str = "<input>",
) -> AssemblyResult:
    """Assemble `source`, reusing the result of an identical recent assembly.

    A cached result raises its warnings again, so a caller sees the same
    warnings whether or not the result came from the cache. A source with a
    search path is never cached: an included file can change between calls.
    """
    key = None if search_path else _cache_key(source, defines, params, abi)
    cached = _cache.get(key) if key is not None else None
    if cached is not None:
        _cache.move_to_end(key)
        cached.reissue_warnings()
        return cached
    result = AssemblySession(
        source, defines=defines, params=params, search_path=search_path, abi=abi, name=name
    )._encode()
    result.reissue_warnings()
    if key is None:
        return result
    _cache[key] = result
    if len(_cache) > _CACHE_SIZE:
        _cache.popitem(last=False)
    return result
