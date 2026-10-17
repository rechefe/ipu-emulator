"""``kernel.toml``: a kernel's whole declaration apart from its program (E06-02).

:func:`load_manifest` reads ``kernel.toml``, checks it against the schema below
and returns a frozen :class:`KernelManifest`; every later stage reads that
view, never the TOML. (The one other reader is ``ipu-as assemble --abi``, which
takes only the ``[abi]`` table to assemble a kernel by hand: ``ipu_as`` cannot
import this package.) An error names the file, the line and the key:
``kernels/elementwise/scale/kernel.toml:14: [abi] binds CR1, a hardwired constant``.

Schema
------

``[kernel]`` (required)
    ``op`` (str, required): the operation, also the shared reference's name.
    ``formats`` (list, required): element formats the kernel runs in, by their
    ``ipu_isa.formats.ElementFormat`` names (``fp32``, ``int8``, ``e1``..``e7``,
    ``int32``). ``asm`` (path): the program, default ``<folder>.asm``.
    ``stages`` (``"file.py:fn"``): a multi-program pipeline (E06-14).
    ``cost``: what routing (E06-16) minimises among the kernels that accept a
    query: a number (negative to prefer a kernel), or an expression in the shape grammar over
    integer parameter and derived names; default 0.

``[params.<name>]`` (or ``<name> = { ... }`` under ``[params]``)
    ``type`` (required): ``int``, ``float``, ``bool`` or ``str``. ``min`` and
    ``max`` (int and float), ``multiple_of`` (int, at least 1), ``choices`` (a
    non-empty list, each in the domain), ``default``, and ``unit``: ``count``
    (default), ``rows`` or ``bytes`` (``bytes`` only for a kernel in
    :data:`BYTE_UNIT_KERNELS`, L16). Floats are finite.

``[inputs.<name>]``, ``[outputs.<name>]``, ``[scratch.<name>]``
    ``shape`` (non-empty list, required): positive ints, or expressions over
    parameter and derived names in E06-03's grammar (``"rows * 4"``,
    ``"ceil(k / 128)"``). ``row``: pin the region's first XMEM row (Q81), an int
    or an expression like a shape's. ``after`` (``"<tensor>"``): place the region
    right after that tensor's in placement order (by default inputs, then
    outputs, then scratch, each in file order), so scratch can come before an
    output; not with ``row``. ``when`` (``"<bool param>"`` or ``"not <bool
    param>"``): the tensor exists only in a case where that holds; otherwise it
    is not placed, generated, passed to the reference or bound.
    ``pack`` (inputs and outputs): a named pack transform (E06-05); an output's
    lays out the stored rows that :func:`ipu_harness.packing.unpack` reads back. ``store`` (outputs):
    ``element`` (default) or ``accumulator``. ``fill`` (scratch): the value the
    region is filled with, default 0: a number, or an expression over
    int and float parameters and derived values in the shape grammar,
    evaluated exactly when the case is planned (``"-1 / channels"``, whole or
    not; a float parameter is its exact binary value) and rounded to the
    format when stored. ``bytes`` (scratch, not with ``fill``):
    ``"file.py:fn"``, or ``"../file.py:fn"`` in the family's folder as a
    generator may be, ``fn(plan)`` returns the region's exact starting bytes,
    format-blind (a lane mask, which no element value can carry). The runner
    writes either through :func:`ipu_harness.packing.region_fill`. Scratch
    regions are placed and filled, never compared. ``range`` (inputs): ``[lo, hi]`` for the generated values in
    every format, or ``{ int8 = [lo, hi], e4 = [lo, hi] }`` per format (E06-08);
    ``lo`` may equal ``hi`` (a constant input).

``[abi]``
    ``<name> = "CR2"`` .. ``"CR15"``, or ``"LR0"`` .. ``"LR15"`` (a preset, E06-06).
    ``CR0`` and ``CR1`` are hardwired constants and cannot be bound, except that
    a tensor pinned at a literal ``row = 0`` may bind ``CR0``: the program reads its base
    through the hardwired 0 and nothing is written. A binding may be a table
    with ``register`` and optional keys: a tensor's may add ``offset``,
    ``{ register = "CR2", offset = -1 }`` for its base row plus an int (in a
    ``unit = "bytes"`` kernel the base and the offset count bytes).
    Any binding may be a table with ``when`` (as a tensor's): it binds
    only in a case where that holds; a conditional tensor's binding takes the
    tensor's condition. Two bindings may share a register only when their
    conditions exclude each other (``bias`` and ``not bias``). A float
    parameter (an element scalar) binds a CR. Every key names a parameter, a
    tensor or a derived value, and every tensor is bound (its base row must
    reach the program), itself or through a bound ``[derived]`` value that
    reads its ``base()``.

``[derived]``
    ``<name> = "<expression>"`` over parameters and earlier derived values,
    in the shape grammar (E06-06). An expression may also read
    ``base(<tensor>)``, the region's base row once placed (in bytes in a
    ``unit = "bytes"`` kernel, as a tensor's binding): such a value, and any
    that reads it, is *placed* (:attr:`KernelManifest.placed`), known only
    when the registers are written, so only ``[abi]`` binds it; a shape, a
    pin or the cost cannot read it. ``w1 = "base(weights) + 1"`` gives one
    region several offset registers, which a ``{ register, offset }``
    binding cannot.

``[cases.<name>]``
    Parameter values: a scalar, or a non-empty list (a sweep axis). Reserved
    keys: ``formats`` (a non-empty subset of the kernel's), ``inputs`` and
    ``expected`` (tensor name to a user file, Q82), ``generator``
    (``"file.py:fn"``, or ``"../file.py:fn"`` in the family's folder, a
    generator its kernels share), ``tier`` (``default`` or ``nightly``, Q73).
    ``[cases.default]`` is required.

``[benchmark]`` (optional, E06-11)
    ``points``: a non-empty list of tables of parameter values, each over the
    default case's, which must take one value each (one run per point and
    format, paired, not swept; a point keeps the default case's generator
    but not its files, and may not change a registry entry's fixed
    parameter); ``per``: an int parameter the report divides cycles by;
    ``max_cycles``: each run's limit (the model's by default). It needs a
    ``[golden]`` reference or op, which each point is held to; no two points
    may be equal. ``python -m ipu_harness.cli KERNEL --benchmark``
    (``bazel run :<kernel> -- --benchmark``) runs them;
    :mod:`ipu_harness.benchmark`.

``[golden]`` (optional)
    ``reference`` (``"file.py:fn"``; default ``reference.py:reference`` when the
    kernel has a ``reference.py``) or ``op`` (a shared reference in
    ``ipu_kernels.reference``), not both; ``tolerance`` (format to a tier name
    of :data:`ipu_harness.tolerance.TIERS`, or a table overriding a tier's
    numbers: ``tier`` and ``why`` (required), ``rtol``, ``atol``, ``ulps``;
    a key is a format some output is compared in: a format the kernel runs in
    for an element output, its accumulator format (FP32, INT32) for an
    accumulator output, so a key no output reads is an error; an override may set only what
    ``compare`` reads for that format: ``ulps`` only for an 8-bit float, and
    then not ``rtol``/``atol``; a format not listed is compared at ``EXACT``).

Every declared name (parameter, tensor, derived value) is an identifier that
does not start with ``_`` and is not ``cr``, ``lr`` or a reserved case key.
Every file path stays inside the kernel directory: ``..`` and absolute paths
are refused (checked lexically). A case ``generator``'s file that is not in the
kernel's folder is its family folder's (``kernels/<family>/gen.py``), as a
``pack`` transform may be the family folder's ``pack.py``: a family's
templates share them. An unknown section or key is an error that
names the closest valid one.

A family ``registry.toml`` entry (E06-17, :mod:`ipu_harness.family`) loads its
template with ``overrides``: ``name``, ``formats`` (a subset), ``params`` (fixed
values: each becomes the parameter's only choice and its value in every case)
and ``cases`` (the template cases the entry runs, ``default`` among them).
"""

from __future__ import annotations

import functools
import math
import re
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from ipu_emu.errors import ConfigurationError
from ipu_harness._codec import limits
from ipu_harness._toml_source import Locator, ManifestError, Reader, decode_error_line, where
from ipu_harness.conditions import Condition
from ipu_harness.expressions import ExpressionError, base_tensor, names_in, placed_reads
from ipu_harness.tolerance import TIERS
from ipu_harness.tolerance import Tier as ToleranceTier
from ipu_isa.formats import FORMATS, ElementFormat
from ipu_isa.registers import REGISTER_DEFINITIONS

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

__all__ = [
    "BYTE_UNIT_KERNELS",
    "TOML_NAME",
    "BenchmarkDecl",
    "CaseDecl",
    "GoldenDecl",
    "KernelManifest",
    "ManifestError",
    "ParamDecl",
    "SCHEMA_KEYS",
    "SECTION_OF",
    "TensorDecl",
    "ZERO_REGISTER",
    "check_abi",
    "load_manifest",
]

TOML_NAME = "kernel.toml"

#: Kernels (``family/name``) allowed ``unit = "bytes"`` parameters (L16). E07
#: lists the legacy kernels it migrates that still count in bytes.
BYTE_UNIT_KERNELS: frozenset[str] = frozenset()

ParamType = Literal["int", "float", "bool", "str"]
Unit = Literal["count", "rows", "bytes"]
Role = Literal["input", "output", "scratch"]
Store = Literal["element", "accumulator"]
Tier = Literal["default", "nightly"]

_PY_TYPES: Mapping[str, tuple[type, ...]] = {
    "int": (int,),
    "float": (int, float),
    "bool": (bool,),
    "str": (str,),
}
_ROLES: Mapping[str, Role] = {"inputs": "input", "outputs": "output", "scratch": "scratch"}
#: The section each role is declared in.
SECTION_OF: Mapping[Role, str] = {role: section for section, role in _ROLES.items()}
_SECTIONS = (
    "kernel",
    "params",
    "inputs",
    "outputs",
    "scratch",
    "abi",
    "derived",
    "cases",
    "golden",
    "benchmark",
)
_BENCHMARK_KEYS = ("points", "per", "max_cycles")
_KERNEL_KEYS = ("op", "formats", "asm", "stages", "cost")
_PARAM_KEYS = ("type", "min", "max", "multiple_of", "choices", "default", "unit")
_TENSOR_KEYS: Mapping[Role, tuple[str, ...]] = {
    "input": ("shape", "row", "after", "when", "pack", "range"),
    "output": ("shape", "row", "after", "when", "pack", "store"),
    "scratch": ("shape", "row", "after", "when", "fill", "bytes"),
}
_CASE_KEYS = ("formats", "inputs", "expected", "generator", "tier")
_GOLDEN_KEYS = ("reference", "op", "tolerance")
_TIER_KEYS = ("tier", "rtol", "atol", "ulps", "why")
_BINDING_KEYS = ("register", "offset", "when")

#: Every table of the schema and the keys it takes, in the module docstring's
#: order: what the ``kernel.toml`` reference page lists (E06-21). A case's
#: other keys are parameter values; ``[abi]`` and ``[derived]`` take names.
SCHEMA_KEYS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "[kernel]": _KERNEL_KEYS,
        "[params.<name>]": _PARAM_KEYS,
        "[inputs.<name>]": _TENSOR_KEYS["input"],
        "[outputs.<name>]": _TENSOR_KEYS["output"],
        "[scratch.<name>]": _TENSOR_KEYS["scratch"],
        "[abi] table binding": _BINDING_KEYS,
        "[cases.<name>] (reserved keys)": _CASE_KEYS,
        "[golden]": _GOLDEN_KEYS,
        "[golden] tolerance override": _TIER_KEYS,
    }
)
#: Names a template or a case reserves (E06-07: cr, lr, format, shapes).
_RESERVED_NAMES = frozenset({"cr", "lr", "format", "shapes", *_CASE_KEYS})


@dataclass(frozen=True)
class ParamDecl:
    """A parameter and its domain.

    Attributes:
        name: Its name in ``[params]``.
        type: ``int``, ``float``, ``bool`` or ``str``.
        min: Smallest value (int and float).
        max: Largest value (int and float).
        multiple_of: Values are multiples of this (int).
        choices: The only values allowed, if given.
        default: The value when a case or query gives none (None: required).
        unit: What an int counts: ``count``, ``rows`` or ``bytes`` (L16).
    """

    name: str
    type: ParamType
    min: int | float | None = None
    max: int | float | None = None
    multiple_of: int | None = None
    choices: tuple[object, ...] | None = None
    default: object | None = None
    unit: Unit = "count"


@dataclass(frozen=True)
class TensorDecl:
    """An input, output or scratch region.

    Attributes:
        name: Its name in its section.
        role: ``input``, ``output`` or ``scratch``.
        shape: Ints, or expressions over parameter and derived names (E06-03).
        row: The first XMEM row, if pinned (Q81): an int, or an expression over
            parameter and derived names.
        after: The tensor this region is placed right after, if it says.
        when: The condition it exists under (``when = "bias"``), if any.
        pack: The named pack transform of an input or output (E06-05).
        store: How an output is stored: ``element`` or ``accumulator``.
        fill: The value a scratch region is filled with: a number, or an
            expression over parameters and derived values, evaluated exactly
            (``"1 / channels"``) and rounded to the format when stored.
        content: ``"file.py:fn"`` giving a scratch region's bytes, if it says.
        value_range: ``[lo, hi]`` per format for an input's generated values
            (``range``, E06-08); a format not listed takes the default.
    """

    name: str
    role: Role
    shape: tuple[int | str, ...]
    row: int | str | None = None
    after: str | None = None
    when: Condition | None = None
    pack: str | None = None
    store: Store = "element"
    fill: int | float | str = 0
    content: str | None = None
    value_range: Mapping[ElementFormat, tuple[float, float]] = field(
        default_factory=lambda: MappingProxyType({})
    )

    @property
    def section(self) -> str:
        """Its section in ``kernel.toml``: ``inputs``, ``outputs`` or ``scratch``."""
        return SECTION_OF[self.role]


@dataclass(frozen=True)
class CaseDecl:
    """A case from ``[cases.<name>]``.

    Attributes:
        name: The case's name.
        values: Parameter values; a tuple is a sweep axis.
        formats: The formats it runs in, if it limits them.
        inputs: User input files, by tensor name (Q82).
        expected: User expected-output files, by tensor name.
        generator: ``"file.py:fn"`` that makes inputs instead.
        tier: ``default`` or ``nightly`` (Q73).
    """

    name: str
    values: Mapping[str, object]
    formats: tuple[ElementFormat, ...] | None = None
    inputs: Mapping[str, Path] = field(default_factory=lambda: MappingProxyType({}))
    expected: Mapping[str, Path] = field(default_factory=lambda: MappingProxyType({}))
    generator: str | None = None
    tier: Tier = "default"


@dataclass(frozen=True)
class BenchmarkDecl:
    """The ``[benchmark]`` table (E06-11): sizes to time the kernel at.

    Attributes:
        points: Parameter values, each over the default case's: one run per
            point and format (paired values, not a sweep).
        per: An int parameter cycles are divided by in the report, if any.
        max_cycles: Each run's cycle limit, if not the model's.
    """

    points: tuple[Mapping[str, object], ...]
    per: str | None = None
    max_cycles: int | None = None


@dataclass(frozen=True)
class GoldenDecl:
    """How outputs are checked.

    Attributes:
        reference: ``"file.py:fn"`` in the kernel directory.
        op: A shared reference in ``ipu_kernels.reference``.
        tolerance: The tier per format it names (E06-10).
    """

    reference: str | None = None
    op: str | None = None
    tolerance: Mapping[ElementFormat, ToleranceTier] = field(
        default_factory=lambda: MappingProxyType({})
    )


@dataclass(frozen=True, eq=False)
class KernelManifest:
    """The checked, frozen view of one ``kernel.toml`` (compared and hashed by identity).

    Attributes:
        name: The folder's name, or the registry entry's (E06-17).
        directory: The kernel's folder.
        asm: The program.
        op: The operation.
        formats: The element formats it runs in.
        params: The parameters, by name.
        tensors: Inputs, outputs and scratch regions, by name.
        abi: Name to ``"CR2"``..``"CR15"`` or ``"LR0"``..``"LR15"`` (``"CR0"`` for a
            tensor pinned at row 0).
        abi_offsets: A tensor's offset from its base row, when ``[abi]`` gives one.
        abi_when: A binding's condition, when it has one (a conditional tensor's
            included).
        derived: Name to expression; a placed one (:attr:`placed`) reads a
            region's base.
        cases: The cases, ``default`` among them.
        golden: How outputs are checked.
        stages: ``"stages.py:run"`` for a multi-program pipeline (E06-14).
        cost: Routing cost: a number, or an expression over integer
            parameters and derived values (E06-16).
        fixed: Parameters a registry entry fixed (E06-17), with their values.
        doc: A registry entry's one-line description, if any.
        registry: The ``registry.toml`` that declares this kernel, for an entry.
    """

    name: str
    directory: Path
    asm: Path
    op: str
    formats: tuple[ElementFormat, ...]
    params: Mapping[str, ParamDecl]
    tensors: Mapping[str, TensorDecl]
    abi: Mapping[str, str]
    derived: Mapping[str, str]
    cases: Mapping[str, CaseDecl]
    golden: GoldenDecl
    stages: str | None = None
    fixed: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))
    cost: int | float | str = 0
    abi_offsets: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))
    abi_when: Mapping[str, Condition] = field(default_factory=lambda: MappingProxyType({}))
    doc: str | None = None
    registry: Path | None = None
    benchmark: BenchmarkDecl | None = None

    @property
    def label(self) -> str:
        """Its Bazel label: ``//kernels/<family>/<name>`` for a folder kernel,
        ``//kernels/<family>:<name>`` for a registry entry."""
        if self.registry is not None:
            return f"//kernels/{self.registry.parent.name}:{self.name}"
        return f"//kernels/{self.directory.parent.name}/{self.directory.name}"

    @functools.cached_property
    def placed(self) -> Mapping[str, frozenset[str]]:
        """The ``[derived]`` values that read ``base(<tensor>)``, directly or
        through another, with the tensors whose bases each reads: known once
        the regions are placed, so only ``[abi]`` reads them."""
        return MappingProxyType(placed_reads(self.derived))

    @property
    def byte_units(self) -> bool:
        """Whether a parameter counts bytes (a legacy kernel, L16)."""
        return any(p.unit == "bytes" for p in self.params.values())


# --- names and expressions ---------------------------------------------------------------


def _check_name(r: Reader, path: tuple[str, ...]) -> None:
    name = path[-1]
    if not name.isidentifier() or name.startswith("_") or name in _RESERVED_NAMES:
        raise r.fail(
            path,
            f"{where(path)}: {name!r} must be an identifier, not start with _, "
            "and not be cr, lr, format, shapes or a reserved case key",
        )


def _check_expression(r: Reader, text: str, known: Iterable[str], path: tuple[str, ...]) -> None:
    """``text`` is in the shape grammar (E06-03) and names only ``known`` values."""
    try:
        names = names_in(text)
    except ExpressionError as error:
        raise r.fail(path, f"{where(path)}: {error}") from None
    unknown = sorted(set(names) - set(known))
    if unknown:
        raise r.fail(path, f"{where(path)} {text!r} names no parameter or derived value: {unknown}")


# --- parameters ---------------------------------------------------------------------------


def _bounds(
    r: Reader, table: Mapping[str, Any], kind: str, path: tuple[str, ...]
) -> dict[str, Any]:
    bounds = {k: r.typed(table[k], (int, float), (*path, k)) for k in ("min", "max") if k in table}
    if bounds and kind not in ("int", "float"):
        raise r.fail(
            (*path, next(iter(bounds))), f"{where(path)}: min and max need type int or float"
        )
    if "min" in bounds and "max" in bounds and bounds["min"] > bounds["max"]:
        raise r.fail(
            (*path, "min"), f"{where(path)}: min {bounds['min']} is above max {bounds['max']}"
        )
    if "multiple_of" in table:
        if kind != "int":
            raise r.fail((*path, "multiple_of"), f'{where(path)}: multiple_of needs type = "int"')
        bounds["multiple_of"] = r.typed(table["multiple_of"], int, (*path, "multiple_of"))
        if bounds["multiple_of"] < 1:
            raise r.fail((*path, "multiple_of"), f"{where(path)}: multiple_of must be at least 1")
    return bounds


def _param(r: Reader, name: str, raw: object, kernel: str) -> ParamDecl:
    path = ("params", name)
    _check_name(r, path)
    table = r.table(raw, path)
    r.known(table, _PARAM_KEYS, path)
    kind = r.require(table, "type", str, path)
    if kind not in _PY_TYPES:
        raise r.fail(
            (*path, "type"), f"{where(path)} type {kind!r} must be one of {', '.join(_PY_TYPES)}"
        )
    decl = ParamDecl(name, kind, **_bounds(r, table, kind, path))
    unit = table.get("unit", "count")
    if unit not in ("count", "rows", "bytes"):
        raise r.fail((*path, "unit"), f"{where(path)} unit {unit!r} must be count, rows or bytes")
    if unit == "bytes" and kernel not in BYTE_UNIT_KERNELS:
        raise r.fail((*path, "unit"), f"{where(path)} counts bytes; a kernel counts rows (L16)")
    decl = replace(decl, unit=unit)
    if "choices" in table:
        listed = r.typed(table["choices"], list, (*path, "choices"))
        if not listed:
            raise r.fail((*path, "choices"), f"{where(path)} choices must not be empty")
        decl = replace(decl, choices=tuple(_value(r, decl, c, (*path, "choices")) for c in listed))
    if "default" in table:
        decl = replace(decl, default=_value(r, decl, table["default"], (*path, "default")))
    return decl


def _value(r: Reader, decl: ParamDecl, raw: object, path: tuple[str, ...]) -> object:
    """``raw`` checked against ``decl``'s domain; an int for a float parameter becomes a float."""
    value: Any = r.typed(raw, _PY_TYPES[decl.type], path)
    if decl.type == "float":
        value = float(value)
    out_of_range = (decl.min is not None and value < decl.min) or (
        decl.max is not None and value > decl.max
    )
    if out_of_range or (decl.multiple_of and value % decl.multiple_of):
        raise r.fail(path, f"{where(path)}: {decl.name} = {value!r} is outside its domain")
    if decl.choices is not None and value not in decl.choices:
        raise r.fail(
            path, f"{where(path)}: {decl.name} = {value!r} is not one of {list(decl.choices)}"
        )
    return value


# --- tensors and the ABI ------------------------------------------------------------------


def _shape(
    r: Reader, raw: object, path: tuple[str, ...], names: Iterable[str]
) -> tuple[int | str, ...]:
    dims = r.typed(raw, list, path)
    if not dims:
        raise r.fail(path, f"{where(path)} must have at least one dimension")
    out: list[int | str] = []
    for dim in dims:
        dim = r.typed(dim, (int, str), path)
        if isinstance(dim, str):
            _check_expression(r, dim, names, path)
        elif dim <= 0:
            raise r.fail(path, f"{where(path)} dimension {dim} must be positive")
        out.append(dim)
    return tuple(out)


def _pair(r: Reader, raw: object, path: tuple[str, ...]) -> tuple[float, float]:
    pair = r.typed(raw, list, path)
    if len(pair) != 2:
        raise r.fail(path, f"{where(path)} must be [lo, hi]")
    lo, hi = (r.typed(v, (int, float), path) for v in pair)
    if lo > hi:
        raise r.fail(path, f"{where(path)} [{lo}, {hi}] is empty: lo must not be above hi")
    return float(lo), float(hi)


def _fits(
    r: Reader, bounds: tuple[float, float], fmt: ElementFormat, path: tuple[str, ...]
) -> None:
    """``bounds`` lie in what ``fmt`` holds, and hold an integer if ``fmt`` is one."""
    lo, hi = bounds
    least, most = limits(fmt)
    if lo < least or hi > most:
        raise r.fail(
            path, f"{where(path)} [{lo:g}, {hi:g}] is outside {fmt.value}'s [{least:g}, {most:g}]"
        )
    if not FORMATS[fmt].is_float and math.ceil(lo) > math.floor(hi):
        raise r.fail(path, f"{where(path)} [{lo:g}, {hi:g}] holds no {fmt.value} value")


def _range(
    r: Reader, raw: object, path: tuple[str, ...], formats: tuple[ElementFormat, ...]
) -> dict[ElementFormat, tuple[float, float]]:
    """``range = [lo, hi]`` for every format, or ``{ int8 = [lo, hi], ... }`` per format."""
    if isinstance(raw, list):
        bounds = _pair(r, raw, path)
        for fmt in formats:
            _fits(r, bounds, fmt, path)
        return dict.fromkeys(formats, bounds)
    out: dict[ElementFormat, tuple[float, float]] = {}
    for key, value in r.table(raw, path).items():
        fmt = r.format(key, (*path, key))
        if fmt not in formats:
            raise r.fail((*path, key), f"range for {key}, a format the kernel does not run in")
        out[fmt] = _pair(r, value, (*path, key))
        _fits(r, out[fmt], fmt, (*path, key))
    return out


def _placed(
    r: Reader,
    decl: TensorDecl,
    table: Mapping[str, Any],
    path: tuple[str, ...],
    names: Iterable[str],
) -> TensorDecl:
    """``decl`` with its ``row`` pin (an int or an expression) or its ``after``."""
    if "row" in table:
        row = r.typed(table["row"], (int, str), (*path, "row"))
        if isinstance(row, str):
            _check_expression(r, row, names, (*path, "row"))
        elif row < 0:
            raise r.fail((*path, "row"), f"{where((*path, 'row'))} {row} must not be negative")
        decl = replace(decl, row=row)
    if "after" in table:
        if "row" in table:
            raise r.fail((*path, "after"), f"{where(path)} is pinned by row; it takes no 'after'")
        decl = replace(decl, after=r.text(table["after"], (*path, "after")))
    return decl


def _condition(r: Reader, raw: object, bools: Iterable[str], path: tuple[str, ...]) -> Condition:
    """``"<bool param>"`` or ``"not <bool param>"``."""
    condition = Condition.parse(r.text(raw, path))
    if condition.param not in bools:
        raise r.fail(path, f"{where(path)} {condition} must name a bool parameter, or 'not' one")
    return condition


def _tensor(
    r: Reader,
    section: str,
    name: str,
    raw: object,
    names: Iterable[str],
    formats: tuple[ElementFormat, ...],
) -> TensorDecl:
    role = _ROLES[section]
    path = (section, name)
    _check_name(r, path)
    table = r.table(raw, path)
    r.known(table, _TENSOR_KEYS[role], path)
    if "shape" not in table:
        raise r.fail(path, f"{where(path)} needs 'shape'")
    decl = TensorDecl(name, role, _shape(r, table["shape"], (*path, "shape"), names))
    decl = _placed(r, decl, table, path, names)
    if "pack" in table:
        decl = replace(decl, pack=r.text(table["pack"], (*path, "pack")))
    if "store" in table:
        store = table["store"]
        if store not in ("element", "accumulator"):
            raise r.fail(
                (*path, "store"),
                f"{where((*path, 'store'))} {store!r} must be element or accumulator",
            )
        decl = replace(decl, store=store)
    if "fill" in table:
        decl = replace(decl, fill=r.typed(table["fill"], (int, float, str), (*path, "fill")))
    if "range" in table:
        ranges = _range(r, table["range"], (*path, "range"), formats)
        decl = replace(decl, value_range=MappingProxyType(ranges))
    return decl


def _binding_problem(
    name: str,
    register: str,
    params: Mapping[str, ParamDecl],
    used: Mapping[str, str],
    *,
    label: str,
    table: str,
) -> str | None:
    """What is wrong with binding ``name`` to ``register`` (already upper case),
    or ``None``; ``label`` names the key in a message, ``table`` the table.
    The caller has checked that ``name`` is the kernel's."""
    m = re.fullmatch(r"(CR|LR)(0|[1-9][0-9]?)", register, re.ASCII)
    banks = {"CR": REGISTER_DEFINITIONS["cr"]["count"], "LR": REGISTER_DEFINITIONS["lr"]["count"]}
    if not m or int(m.group(2)) >= banks[m.group(1)]:
        return f"{label} = {register!r} is no CR or LR register"
    if m.group(1) == "CR" and int(m.group(2)) < 2:
        return f"{table} binds {register}, a hardwired constant"
    if register in used:
        return f"{table} {name} and {used[register]} both bind {register}"
    if name in params and params[name].type == "str":
        return f"{table} {name} is a str parameter; a register holds a number"
    if register.startswith("LR") and name in params and params[name].type == "float":
        return f"{table} {name} is an element scalar (a float); it binds a CR, not {register}"
    return None


#: The hardwired zero a tensor pinned at row 0 may be read through.
ZERO_REGISTER = "CR0"


def _register(
    r: Reader, name: str, raw: object, tensor: TensorDecl | None, params: Mapping[str, ParamDecl]
) -> str:
    """The register of one binding, in upper case: any but CR0 and CR1 by
    :func:`_binding_problem`'s rules, or CR0 for a tensor pinned at row 0.
    Sharing a register is :func:`_abi`'s check (conditions may exclude)."""
    path = ("abi", name)
    text = str(r.typed(raw, str, path))
    register = text.upper()
    if register == ZERO_REGISTER and tensor is not None:
        if tensor.row != 0:
            raise r.fail(path, f"[abi] {name} is read through CR0 (always 0); pin it with row = 0")
        return register
    problem = _binding_problem(name, register, params, {}, label=where(path), table="[abi]")
    if problem is not None:
        # The kernel.toml spelling, not the upper-case one.
        raise r.fail(path, problem.replace(repr(register), repr(text), 1))
    return register


@dataclass(frozen=True)
class _Binding:
    register: str
    offset: int
    when: Condition | None


def _binding(
    r: Reader,
    name: str,
    raw: object,
    tensor: TensorDecl | None,
    params: Mapping[str, ParamDecl],
) -> _Binding:
    """A register, or ``{ register = ..., offset = n, when = ... }``."""
    path = ("abi", name)
    when = tensor.when if tensor is not None else None
    if not isinstance(raw, dict):
        return _Binding(_register(r, name, raw, tensor, params), 0, when)
    r.known(raw, _BINDING_KEYS, path)
    if "register" not in raw:
        raise r.fail(path, f"{where(path)} needs 'register'")
    register = _register(r, name, raw["register"], tensor, params)
    if "offset" in raw and tensor is None:
        raise r.fail((*path, "offset"), f"[abi] {name}: only a tensor's binding takes an offset")
    offset = r.typed(raw.get("offset", 0), int, (*path, "offset"))
    if register == ZERO_REGISTER and offset:
        raise r.fail((*path, "offset"), f"[abi] {name} is read through CR0; it takes no offset")
    if "when" in raw:
        bools = [n for n, p in params.items() if p.type == "bool"]
        own = _condition(r, raw["when"], bools, (*path, "when"))
        if tensor is not None and own != when:
            # An always-present tensor must always be bound, and a conditional
            # one exactly when it exists.
            raise r.fail(
                (*path, "when"),
                f"[abi] {name} takes its tensor's condition ({when or 'always'}), not {own}",
            )
        when = own
    return _Binding(register, offset, when)


def _abi(
    r: Reader,
    raw: object,
    params: Mapping[str, ParamDecl],
    names: set[str],
    tensors: Mapping[str, TensorDecl],
    placed: Mapping[str, frozenset[str]],
) -> tuple[Mapping[str, str], Mapping[str, int], Mapping[str, Condition]]:
    table = r.table(raw, ("abi",))
    abi: dict[str, str] = {}
    offsets: dict[str, int] = {}
    conditions: dict[str, Condition] = {}
    used: dict[str, list[str]] = {}
    for name, value in table.items():
        path = ("abi", name)
        if name not in names:
            raise r.fail(path, f"{where(path)} names no parameter, tensor or derived value")
        binding = _binding(r, name, value, tensors.get(name), params)
        register = binding.register
        if binding.offset:
            offsets[name] = binding.offset
        if binding.when is not None:
            conditions[name] = binding.when
        for other in used.get(register, []):
            if binding.when is None or not binding.when.excludes(conditions.get(other)):
                raise r.fail(path, f"[abi] {name} and {other} both bind {register}")
        used.setdefault(register, []).append(name)
        abi[name] = register
    # A bound placed value that reads base(x) carries x's base to the program
    # too (a region bound at a parameter-dependent offset, E07-15).
    reached = {tensor for name, reads in placed.items() if name in abi for tensor in reads}
    for tensor in tensors.values():
        if tensor.name not in abi and tensor.name not in reached:
            raise r.fail(
                (tensor.section, tensor.name),
                f"[abi] does not bind tensor {tensor.name!r}; its base row must reach the program",
            )
    return MappingProxyType(abi), MappingProxyType(offsets), MappingProxyType(conditions)


def check_abi(manifest: KernelManifest, abi: Mapping[str, str], *, where: str) -> Mapping[str, str]:
    """``abi`` checked by ``[abi]``'s rules, for a table that is not the
    kernel's own (a stage's, E06-14): each name is the kernel's parameter,
    tensor or derived value, each register a CR (not CR0 or CR1) or LR bound
    once, no ``str`` parameter, no float on an LR. Unlike ``[abi]``, it need
    not bind every tensor.

    Args:
        manifest: The kernel whose names ``abi`` binds.
        abi: Names to registers (``{"x": "CR2"}``; any case).
        where: What the table is, for the error (``"stage copy.asm"``).

    Returns:
        The table, registers in upper case.

    Raises:
        ConfigurationError: A binding breaks a rule.
    """
    names = {*manifest.params, *manifest.tensors, *manifest.derived}
    checked: dict[str, str] = {}
    for name, value in abi.items():
        if name not in names:
            raise ConfigurationError(
                f"{manifest.name}: {where} abi {name!r} names no parameter, tensor or derived value"
            )
        register = value.upper() if isinstance(value, str) else str(value)
        problem = _binding_problem(
            name,
            register,
            manifest.params,
            {v: k for k, v in checked.items()},
            label=f"{where} abi {name!r}",
            table=f"{where}'s abi",
        )
        if problem is not None:
            # The value as the stage wrote it, not the upper-case one.
            shown = problem.replace(repr(register), repr(value), 1)
            raise ConfigurationError(f"{manifest.name}: {shown}")
        checked[name] = register
    return MappingProxyType(checked)


# --- cases and golden ---------------------------------------------------------------------


def _files(
    r: Reader, directory: Path, raw: object, path: tuple[str, ...], allowed: Iterable[str]
) -> Mapping[str, Path]:
    table = r.table(raw, path)
    valid = tuple(allowed)
    out = {}
    for tensor, value in table.items():
        if tensor not in valid:
            raise r.fail(
                (*path, tensor), f"{where(path)} {tensor!r} is not one of the tensors {list(valid)}"
            )
        out[tensor] = r.inside(directory, r.typed(value, str, (*path, tensor)), (*path, tensor))
    return MappingProxyType(out)


def _case_values(
    r: Reader, m: KernelManifest, path: tuple[str, ...], table: Mapping[str, Any]
) -> dict[str, object]:
    """The case's parameter values: a scalar, or a tuple (a sweep axis). A fixed
    parameter takes its fixed value (E06-17)."""
    values: dict[str, object] = {}
    for key, value in table.items():
        if key not in m.params:
            continue
        items = value if isinstance(value, list) else [value]
        if not items:
            raise r.fail((*path, key), f"{where((*path, key))} sweeps over no values")
        if key in m.fixed:
            continue  # the registry entry's value replaces the template's
        checked = [_value(r, m.params[key], v, (*path, key)) for v in items]
        values[key] = tuple(checked) if isinstance(value, list) else checked[0]
    values.update(m.fixed)
    return values


def _case(r: Reader, m: KernelManifest, name: str, raw: object) -> CaseDecl:
    path = ("cases", name)
    table = r.table(raw, path)
    r.known(table, (*m.params, *_CASE_KEYS), path)
    case = CaseDecl(name, MappingProxyType(_case_values(r, m, path, table)))
    if "formats" in table:
        listed = r.typed(table["formats"], list, (*path, "formats"))
        formats = tuple(r.format(f, (*path, "formats")) for f in listed)
        if not formats:
            raise r.fail((*path, "formats"), f"{where(path)} formats must not be empty")
        extra = [f.value for f in formats if f not in m.formats]
        if extra:
            raise r.fail((*path, "formats"), f"{where(path)} formats {extra} are not the kernel's")
        case = replace(case, formats=formats)
    if "inputs" in table:
        inputs = [t.name for t in m.tensors.values() if t.role == "input"]
        case = replace(
            case, inputs=_files(r, m.directory, table["inputs"], (*path, "inputs"), inputs)
        )
    if "expected" in table:
        outputs = [t.name for t in m.tensors.values() if t.role == "output"]
        expected = _files(r, m.directory, table["expected"], (*path, "expected"), outputs)
        case = replace(case, expected=expected)
    if "generator" in table:
        case = replace(
            case,
            generator=r.shared_function(m.directory, table["generator"], (*path, "generator")),
        )
    if "tier" in table:
        tier = table["tier"]
        if tier not in ("default", "nightly"):
            raise r.fail(
                (*path, "tier"), f"{where((*path, 'tier'))} {tier!r} must be default or nightly"
            )
        case = replace(case, tier=tier)
    return case


def _named_tier(r: Reader, raw: object, path: tuple[str, ...]) -> ToleranceTier:
    name = r.text(raw, path)
    if name not in TIERS:
        raise r.fail(path, f"{where(path)} {name!r} is no tier; tiers: {', '.join(TIERS)}")
    return TIERS[name]


def _tier(r: Reader, raw: object, fmt: ElementFormat, path: tuple[str, ...]) -> ToleranceTier:
    """A tier by name, or a table overriding one's numbers with a ``why``.

    An override may set only the numbers ``compare`` reads for ``fmt``:
    ``ulps`` only for an 8-bit float format, and there not ``rtol`` or ``atol``
    when the tier counts codes.
    """
    if not isinstance(raw, dict):
        return _named_tier(r, raw, path)
    r.known(raw, _TIER_KEYS, path)
    if "tier" not in raw or "why" not in raw:
        raise r.fail(path, f"{where(path)} overrides a tier: it needs 'tier' and a 'why'")
    base = _named_tier(r, raw["tier"], (*path, "tier"))
    fp8 = FORMATS[fmt].is_float and FORMATS[fmt].element_bits == 8
    unread = ["ulps"] if not fp8 else []
    if fp8 and (base.ulps is not None or "ulps" in raw):
        unread = ["rtol", "atol"]
    for key in unread:
        if key in raw:
            raise r.fail(
                (*path, key), f"{where((*path, key))} is never read for {fmt.value} at this tier"
            )
    rtol, atol = (_non_negative(r, raw, key, (int, float), path) for key in ("rtol", "atol"))
    ulps = _non_negative(r, raw, "ulps", int, path)
    return ToleranceTier(
        base.name,
        base.rtol if rtol is None else float(rtol),
        base.atol if atol is None else float(atol),
        base.ulps if ulps is None else int(ulps),
        r.text(raw["why"], (*path, "why")),
    )


def _non_negative(
    r: Reader,
    table: Mapping[str, Any],
    key: str,
    kind: type | tuple[type, ...],
    path: tuple[str, ...],
) -> float | None:
    """``table[key]`` when given: a number of ``kind`` that is not negative."""
    if key not in table:
        return None
    value = r.typed(table[key], kind, (*path, key))
    if value < 0:
        raise r.fail((*path, key), f"{where((*path, key))} {value} must not be negative")
    return float(value)


def _read_formats(
    formats: tuple[ElementFormat, ...], tensors: Mapping[str, TensorDecl]
) -> set[ElementFormat]:
    """The formats some output is compared in: a run's format for an element
    output, its accumulator format (FP32, INT32) for an accumulator output
    (as ``packing.stored_format``, which this module cannot import)."""
    read = set()
    for tensor in tensors.values():
        if tensor.role != "output":
            continue
        for fmt in formats:
            accumulated = ElementFormat(FORMATS[fmt].accumulator.value)
            read.add(accumulated if tensor.store == "accumulator" else fmt)
    return read


def _golden(
    r: Reader,
    directory: Path,
    raw: object,
    formats: tuple[ElementFormat, ...],
    tensors: Mapping[str, TensorDecl],
) -> GoldenDecl:
    table = r.table(raw, ("golden",))
    r.known(table, _GOLDEN_KEYS, ("golden",))
    if "reference" in table and "op" in table:
        raise r.fail(("golden", "op"), "[golden] names both a reference and an op; keep one")
    reference = None
    if "reference" in table:
        reference = r.function(directory, table["reference"], ("golden", "reference"))
    elif "op" not in table and (directory / "reference.py").is_file():
        reference = "reference.py:reference"
    op = r.text(table["op"], ("golden", "op")) if "op" in table else None
    tolerance: dict[ElementFormat, ToleranceTier] = {}
    if "tolerance" in table:
        tiers = r.table(table["tolerance"], ("golden", "tolerance"))
        # An accumulator output is compared in its accumulator format (FP32 or INT32).
        accumulated = {ElementFormat(FORMATS[f].accumulator.value) for f in formats}
        read = _read_formats(formats, tensors)
        for fmt, tier in tiers.items():
            path = ("golden", "tolerance", fmt)
            key = r.format(fmt, path)
            if key not in formats and key not in accumulated:
                raise r.fail(
                    path,
                    f"tolerance for {fmt}, a format the kernel does not run in or accumulate in",
                )
            if key not in read:
                compared = ", ".join(sorted(f.value for f in read)) or "none"
                raise r.fail(
                    path,
                    f"tolerance for {fmt} is never read: no output is compared in {fmt} "
                    f"(an accumulator output is compared in its accumulator format; "
                    f"outputs are compared in: {compared})",
                )
            tolerance[key] = _tier(r, tier, key, path)
    return GoldenDecl(reference, op, MappingProxyType(tolerance))


# --- the loader ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _KernelTable:
    op: str
    formats: tuple[ElementFormat, ...]
    asm: Path
    stages: str | None
    cost: int | float | str


def _kernel(r: Reader, directory: Path, raw: object) -> _KernelTable:
    table = r.table(raw, ("kernel",))
    r.known(table, _KERNEL_KEYS, ("kernel",))
    if "op" not in table:
        raise r.fail(("kernel",), "[kernel] needs 'op'")
    op = r.text(table["op"], ("kernel", "op"))
    listed = r.require(table, "formats", list, ("kernel",))
    formats = tuple(r.format(f, ("kernel", "formats")) for f in listed)
    if not formats or len(set(formats)) != len(formats):
        raise r.fail(
            ("kernel", "formats"), "[kernel] formats must list each format once, at least one"
        )
    asm_name = r.typed(table.get("asm", f"{directory.name}.asm"), str, ("kernel", "asm"))
    asm = r.inside(directory, asm_name, ("kernel", "asm"))
    if not asm.is_file():
        raise r.fail(("kernel", "asm"), f"the program {asm_name} is not in the kernel directory")
    stages = (
        r.function(directory, table["stages"], ("kernel", "stages")) if "stages" in table else None
    )
    cost = r.typed(table.get("cost", 0), (int, float, str), ("kernel", "cost"))
    return _KernelTable(op, formats, asm, stages, cost)


@dataclass(frozen=True)
class _Overrides:
    name: str | None
    formats: tuple[ElementFormat, ...]
    fixed: Mapping[str, object]
    cases: tuple[str, ...] | None = None


def _overrides(
    toml: Path,
    params: dict[str, ParamDecl],
    formats: tuple[ElementFormat, ...],
    overrides: Mapping[str, Any],
    source: str,
) -> _Overrides:
    """Apply a registry entry; fixed parameters narrow ``params``. Errors name ``source``."""

    def fail(message: str) -> ManifestError:
        return ManifestError(f"{source}: override of {toml}: {message}")

    unknown = sorted(set(overrides) - {"name", "params", "formats", "cases"})
    if unknown:
        raise fail(f"unknown overrides {unknown}; valid: name, formats, params, cases")
    name = overrides.get("name")
    if name is not None and (not isinstance(name, str) or not name.isidentifier()):
        raise fail(f"name {name!r} must be an identifier")
    if "formats" in overrides:
        listed = overrides["formats"]
        valid = {f.value for f in formats}
        if not isinstance(listed, list) or not listed or any(f not in valid for f in listed):
            raise fail(f"formats {listed!r} must be a non-empty subset of {sorted(valid)}")
        formats = tuple(ElementFormat(f) for f in listed)
    fixed = _fixed(toml, params, overrides.get("params", {}), fail)
    cases = overrides.get("cases")
    if cases is not None and not _case_names(cases):
        raise fail(f"cases {cases!r} must list template cases once each, default among them")
    return _Overrides(
        name, formats, MappingProxyType(fixed), None if cases is None else tuple(cases)
    )


def _fixed(
    toml: Path,
    params: dict[str, ParamDecl],
    raw: object,
    fail: Callable[[str], ManifestError],
) -> dict[str, object]:
    """An entry's fixed values, each made its parameter's only choice in ``params``."""
    if not isinstance(raw, Mapping):
        raise fail(f"params {raw!r} must be a table")
    fixed: dict[str, object] = {}
    probe = Reader(toml, Locator(""))
    for pname, value in raw.items():
        if pname not in params:
            raise fail(f"params.{pname} is no parameter of the template")
        try:
            fixed[pname] = _value(probe, params[pname], value, ("params", pname))
        except ManifestError as error:
            raise fail(str(error).split(": ", 1)[-1]) from None
        params[pname] = replace(params[pname], choices=(fixed[pname],), default=fixed[pname])
    return fixed


def _case_names(cases: object) -> bool:
    """Whether an entry's ``cases`` names cases once each, ``default`` among them."""
    return (
        isinstance(cases, list)
        and all(isinstance(c, str) for c in cases)
        and "default" in cases
        and len(set(cases)) == len(cases)
    )


def _derived(r: Reader, raw: object, params: Mapping[str, ParamDecl]) -> dict[str, str]:
    derived: dict[str, str] = {}
    for name, expression in r.table(raw, ("derived",)).items():
        path = ("derived", name)
        _check_name(r, path)
        if name in params:
            raise r.fail(path, f"[derived] {name} is also a parameter")
        derived[name] = r.text(expression, path)
        bases = [n for n in _names(r, derived[name], path) if base_tensor(n) is not None]
        earlier = [d for d in derived if d != name]  # not itself: it would never resolve
        _check_expression(r, derived[name], (*params, *earlier, *bases), path)
    return derived


def _names(r: Reader, text: str, path: tuple[str, ...]) -> list[str]:
    try:
        return names_in(text)
    except ExpressionError as error:
        raise r.fail(path, f"{where(path)}: {error}") from None


def _declared_tensors(r: Reader, data: Mapping[str, Any]) -> list[str]:
    """Every tensor name the file declares, before its tables are read."""
    return [name for section in _ROLES for name in r.table(data.get(section, {}), (section,))]


def _placed_derived(
    r: Reader, derived: Mapping[str, str], tensors: Iterable[str]
) -> Mapping[str, frozenset[str]]:
    """The derived values that read a region's base (``placed_reads``), once
    each ``base(x)`` is checked to name a tensor."""
    declared = set(tensors)
    for name, expression in derived.items():
        path = ("derived", name)
        for read in _names(r, expression, path):
            tensor = base_tensor(read)
            if tensor is not None and tensor not in declared:
                raise r.fail(path, f"{where(path)} {expression!r}: {read} names no tensor")
    return placed_reads(derived)


def _no_placed_reads(
    r: Reader, placed: Iterable[str], tensors: Mapping[str, TensorDecl], cost: object
) -> None:
    """A shape, a pin or the cost reads no placed value: it is known only once
    the regions are, after them."""
    late = set(placed)
    reading: list[tuple[tuple[str, ...], str]] = [
        ((tensor.section, tensor.name, key), text)
        for tensor in tensors.values()
        for key, texts in (
            ("shape", tensor.shape),
            ("row", (tensor.row,)),
            ("fill", (tensor.fill,)),
        )
        for text in texts
        if isinstance(text, str)
    ]
    if isinstance(cost, str):
        reading.append((("kernel", "cost"), cost))
    for path, text in reading:
        read = sorted(late & set(_names(r, text, path)))
        if read:
            raise r.fail(
                path,
                f"{where(path)} {text!r} reads {read}, which read a region's base: "
                "only [abi] may read such a value",
            )


def _placed_conditions(
    r: Reader,
    placed: Mapping[str, frozenset[str]],
    tensors: Mapping[str, TensorDecl],
    abi_when: Mapping[str, Condition],
    abi: Mapping[str, str],
) -> None:
    """A placed value's binding takes the condition of the conditional tensors
    it reads, as a tensor's binding takes its tensor's: it is bound exactly
    when they exist."""
    for name, read in placed.items():
        if name not in abi:
            continue
        conditions = {tensors[t].when for t in read if tensors[t].when is not None}
        if len(conditions) > 1:
            raise r.fail(("abi", name), f"[abi] {name} reads tensors under different conditions")
        if conditions and abi_when.get(name) not in conditions:
            want = next(iter(conditions))
            raise r.fail(
                ("abi", name),
                f"[abi] {name} reads the base of a tensor that exists when {want}; "
                f"bind it with when = {str(want)!r}",
            )


def _tensors(
    r: Reader,
    data: Mapping[str, Any],
    scalars: tuple[str, ...],
    formats: tuple[ElementFormat, ...],
    bools: tuple[str, ...],
    directory: Path,
) -> dict[str, TensorDecl]:
    tensors: dict[str, TensorDecl] = {}
    for section in _ROLES:
        for name, raw in r.table(data.get(section, {}), (section,)).items():
            if name in tensors or name in scalars:
                raise r.fail((section, name), f"{name!r} is declared twice")
            decl = _tensor(r, section, name, raw, scalars, formats)
            if isinstance(raw, dict) and "when" in raw:
                when = _condition(r, raw["when"], bools, (section, name, "when"))
                decl = replace(decl, when=when)
            if isinstance(raw, dict) and "bytes" in raw:
                decl = replace(decl, content=_content(r, directory, raw, (section, name)))
            tensors[name] = decl
    for tensor in tensors.values():
        _check_after(r, tensor, tensors)
    return tensors


def _content(r: Reader, directory: Path, table: Mapping[str, Any], path: tuple[str, ...]) -> str:
    """A scratch region's ``bytes = "file.py:fn"`` (or ``"../file.py:fn"``, the family's)."""
    if "fill" in table:
        raise r.fail((*path, "bytes"), f"{where(path)} takes 'fill' or 'bytes', not both")
    return r.shared_function(directory, table["bytes"], (*path, "bytes"))


def _check_after(r: Reader, tensor: TensorDecl, tensors: Mapping[str, TensorDecl]) -> None:
    """``after`` names another tensor, and following it never comes back."""
    path = (tensor.section, tensor.name, "after")
    seen = {tensor.name}
    current = tensor
    while current.after is not None:
        if current.after not in tensors:
            raise r.fail(path, f"{where(path)} {current.after!r} is no tensor")
        if current.after in seen:
            raise r.fail(path, f"{where(path)}: after forms a cycle through {current.after!r}")
        seen.add(current.after)
        current = tensors[current.after]


def _read(toml: Path) -> tuple[Reader, dict[str, Any]]:
    try:
        text = toml.read_text(encoding="utf-8")
    except OSError as error:
        raise ManifestError(f"{toml}:1: cannot read it: {error.strerror}") from None
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise ManifestError(f"{toml}:{decode_error_line(str(error))}: {error}") from None
    return Reader(toml, Locator(text)), data


def load_manifest(
    directory: Path, *, overrides: Mapping[str, Any] | None = None, source: str = "overrides"
) -> KernelManifest:
    """Read and check ``directory/kernel.toml``; the frozen view of it.

    Args:
        directory: The kernel's folder (holding ``kernel.toml`` and its program).
        overrides: From a family ``registry.toml`` entry (E06-17): ``name``,
            ``formats`` (a subset) and ``params`` (fixed values).
        source: Where ``overrides`` come from, for their errors.

    Returns:
        The manifest.

    Raises:
        ManifestError: The file is missing or breaks the schema; the message names the line.
    """
    toml = directory / TOML_NAME
    r, data = _read(toml)
    r.known(data, _SECTIONS, ())
    if "kernel" not in data:
        raise r.fail((), f"{TOML_NAME} needs a [kernel] section")
    kernel = _kernel(r, directory, data["kernel"])
    family_name = f"{directory.parent.name}/{directory.name}"
    raw_params = r.table(data.get("params", {}), ("params",))
    params = {name: _param(r, name, raw, family_name) for name, raw in raw_params.items()}
    entry = _overrides(toml, params, kernel.formats, overrides or {}, source)
    derived = _derived(r, data.get("derived", {}), params)
    # A placed derived value (it reads base(x)) is known only once the regions
    # are: a shape, a pin or the cost cannot read it, only [abi].
    placed = _placed_derived(r, derived, _declared_tensors(r, data))
    scalars = (*params, *derived)
    if isinstance(kernel.cost, str):
        integers = [n for n, p in params.items() if p.type == "int"]
        _check_expression(r, kernel.cost, (*integers, *derived), ("kernel", "cost"))
    bools = tuple(n for n, p in params.items() if p.type == "bool")
    tensors = _tensors(r, data, scalars, kernel.formats, bools, directory)
    numbers = [n for n, p in params.items() if p.type in ("int", "float")]
    for tensor in tensors.values():
        if isinstance(tensor.fill, str):
            path = (tensor.section, tensor.name, "fill")
            _check_expression(r, tensor.fill, (*numbers, *derived), path)
    _no_placed_reads(r, placed, tensors, kernel.cost)
    abi, abi_offsets, abi_when = _abi(
        r, data.get("abi", {}), params, {*scalars, *tensors}, tensors, placed
    )
    _placed_conditions(r, placed, tensors, abi_when, abi)
    manifest = KernelManifest(
        name=entry.name or directory.name,
        directory=directory,
        asm=kernel.asm,
        op=kernel.op,
        formats=entry.formats,
        params=MappingProxyType(params),
        tensors=MappingProxyType(tensors),
        abi=abi,
        abi_offsets=abi_offsets,
        abi_when=abi_when,
        derived=MappingProxyType(derived),
        cases=MappingProxyType({}),
        golden=_golden(r, directory, data.get("golden", {}), kernel.formats, tensors),
        stages=kernel.stages,
        fixed=entry.fixed,
        cost=kernel.cost,
    )
    raw_cases = r.table(data.get("cases", {}), ("cases",))
    if "default" not in raw_cases:
        raise r.fail(("cases",), f"{TOML_NAME} needs [cases.default]")
    if entry.cases is not None:
        missing = sorted(set(entry.cases) - set(raw_cases))
        if missing:
            raise ManifestError(
                f"{source}: override of {toml}: cases {missing} are not the template's"
            )
        raw_cases = {name: raw_cases[name] for name in entry.cases}
    cases = {name: _case(r, manifest, name, raw) for name, raw in raw_cases.items()}
    manifest = replace(manifest, cases=MappingProxyType(cases))
    if "benchmark" in data:
        manifest = replace(manifest, benchmark=_benchmark(r, manifest, data["benchmark"]))
    return manifest


def _point(r: Reader, m: KernelManifest, i: int, raw: object) -> Mapping[str, object]:
    """Benchmark point ``i``: one value per parameter, none moving a fixed one."""
    path = ("benchmark", "points", str(i))
    values = r.table(raw, path)
    r.known(values, tuple(m.params), path)
    sweeps = [k for k, v in values.items() if isinstance(v, list)]
    if sweeps:
        raise r.fail(path, f"[benchmark] point {i}: {sweeps[0]} is one value, not a list")
    moved = [k for k, v in values.items() if k in m.fixed and v != m.fixed[k]]
    if moved:
        raise r.fail(
            path,
            f"[benchmark] point {i}: {moved[0]} is fixed at {m.fixed[moved[0]]!r} "
            "by the registry entry",
        )
    return MappingProxyType(_case_values(r, m, path, values))


def _benchmark(r: Reader, m: KernelManifest, raw: object) -> BenchmarkDecl:
    path = ("benchmark",)
    table = r.table(raw, path)
    r.known(table, _BENCHMARK_KEYS, path)
    if m.golden.reference is None and m.golden.op is None:
        # A point's shapes are its own, so no expected file fits it: without a
        # reference a point would compare nothing and pass.
        raise r.fail(path, f"{where(path)} needs a [golden] reference or op to hold each point to")
    default = m.cases["default"].values
    if any(isinstance(v, tuple) for v in default.values()):
        raise r.fail(
            path, f"{where(path)} needs a one-value [cases.default] to put its points over"
        )
    listed = r.typed(table.get("points", []), list, (*path, "points"))
    if not listed:
        raise r.fail((*path, "points"), f"{where(path)} needs a non-empty points list")
    points: list[Mapping[str, object]] = []
    for i, raw_point in enumerate(listed):
        point = _point(r, m, i, raw_point)
        if point in points:
            # Its run, its row and its history name would be an earlier one's.
            raise r.fail(
                (*path, "points", str(i)),
                f"[benchmark] point {i} repeats point {points.index(point)}",
            )
        points.append(point)
    per = table.get("per")
    if per is not None and r.typed(per, str, (*path, "per")) not in m.params:
        raise r.fail((*path, "per"), f"{where(path)} per {per!r} is no int parameter")
    if per is not None and m.params[per].type != "int":
        raise r.fail((*path, "per"), f"{where(path)} per {per!r} is no int parameter")
    max_cycles = table.get("max_cycles")
    if max_cycles is not None and r.typed(max_cycles, int, (*path, "max_cycles")) < 1:
        raise r.fail((*path, "max_cycles"), f"{where(path)} max_cycles must be at least 1")
    return BenchmarkDecl(tuple(points), per, max_cycles)
