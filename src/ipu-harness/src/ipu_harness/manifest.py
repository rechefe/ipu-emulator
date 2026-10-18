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
    :data:`BYTE_UNIT_KERNELS`, L16). Floats are finite. ``query_required``
    (bool, default false): a routing query that leaves the parameter out is
    refused at this kernel, naming it, whatever its default; a case takes its
    own or the default value as before.

``[inputs.<name>]``, ``[outputs.<name>]``, ``[scratch.<name>]``
    ``shape`` (non-empty list, required): positive ints, or expressions over
    parameter and derived names in E06-03's grammar (``"rows * 4"``,
    ``"ceil(k / 128)"``). ``row``: pin the region's first XMEM row (Q81), an int
    or an expression like a shape's, which may also read ``row_bytes``, the
    bytes of one row in the case's format (``"131072 / row_bytes"`` pins the
    region at byte 0x20000 in every format: a legacy layout). ``after`` (``"<tensor>"``): place the region
    right after that tensor's in placement order (by default inputs, then
    outputs, then scratch, each in file order), so scratch can come before an
    output; not with ``row``. ``when`` (``"<bool param>"`` or ``"not <bool
    param>"``, such terms joined by ``and``, and those joined by ``or``:
    ``"has_bias or apply_relu"``, ``"not has_bias and not apply_relu"``):
    the tensor exists only in a case where that holds; otherwise it
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
    conditions exclude each other (``bias`` and ``not bias``; a condition with
    ``or`` excludes another only when every term of one contradicts every term
    of the other), or when both are tensors pinned at literal rows that put the
    same value in it (``acc`` at ``row = 8`` and ``relu = { register = "CR7",
    offset = -12 }`` at ``row = 20``: the program reaches relu from acc's
    base). A float
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
    binding cannot. A value that reads ``row_bytes`` (the bytes of one XMEM
    row in the case's format) is placed too: ``"512 / row_bytes"`` is the
    rows one accumulator vector spans (4 in an 8-bit format, 1 in FP32 and
    INT32).

``[cases.<name>]``
    Parameter values: a scalar, or a non-empty list (a sweep axis). Reserved
    keys: ``formats`` (a non-empty subset of the kernel's), ``inputs`` and
    ``expected`` (tensor name to a user file, Q82; an ``expected`` may instead
    be ``"file.py:fn"``, or ``"../file.py:fn"`` in the family's folder, whose
    ``fn(plan)`` returns the output's values in the case's format: a golden
    kept in another encoding, such as ``fully_connected``'s ``.bin``), ``generator``
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
values: each becomes the parameter's only choice and its value in every case;
a default the template gives stays, so a query that leaves the parameter out
means that default, else the fixed value stands in)
and ``cases`` (the template cases the entry runs, ``default`` among them).
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ipu_harness._manifest_abi import ZERO_REGISTER, check_abi, check_placed_conditions, read_abi
from ipu_harness._manifest_cases import read_benchmark, read_case
from ipu_harness._manifest_decls import (
    BINDING_KEYS,
    CASE_KEYS,
    GOLDEN_KEYS,
    KERNEL_KEYS,
    PARAM_KEYS,
    SECTION_OF,
    TENSOR_KEYS,
    TIER_KEYS,
    BenchmarkDecl,
    CaseDecl,
    GoldenDecl,
    KernelManifest,
    ParamDecl,
    ParamType,
    Role,
    Store,
    TensorDecl,
    Tier,
    Unit,
)
from ipu_harness._manifest_golden import read_golden
from ipu_harness._manifest_params import check_expression, checked_value, read_derived, read_param
from ipu_harness._manifest_tensors import (
    check_no_placed_reads,
    declared_tensors,
    placed_derived,
    read_tensors,
)
from ipu_harness._toml_source import Locator, ManifestError, Reader, decode_error_line
from ipu_isa.formats import ElementFormat

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
    "ParamType",
    "Role",
    "SCHEMA_KEYS",
    "SECTION_OF",
    "Store",
    "TensorDecl",
    "Tier",
    "Unit",
    "ZERO_REGISTER",
    "check_abi",
    "load_manifest",
]

TOML_NAME = "kernel.toml"

#: Kernels (``family/name``) allowed ``unit = "bytes"`` parameters (L16). E07
#: lists the legacy kernels it migrates that still count in bytes.
BYTE_UNIT_KERNELS: frozenset[str] = frozenset()

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

#: Every table of the schema and the keys it takes, in the module docstring's
#: order: what the ``kernel.toml`` reference page lists (E06-21). A case's
#: other keys are parameter values; ``[abi]`` and ``[derived]`` take names.
SCHEMA_KEYS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "[kernel]": KERNEL_KEYS,
        "[params.<name>]": PARAM_KEYS,
        "[inputs.<name>]": TENSOR_KEYS["input"],
        "[outputs.<name>]": TENSOR_KEYS["output"],
        "[scratch.<name>]": TENSOR_KEYS["scratch"],
        "[abi] table binding": BINDING_KEYS,
        "[cases.<name>] (reserved keys)": CASE_KEYS,
        "[golden]": GOLDEN_KEYS,
        "[golden] tolerance override": TIER_KEYS,
    }
)


@dataclass(frozen=True)
class _KernelTable:
    op: str
    formats: tuple[ElementFormat, ...]
    asm: Path
    stages: str | None
    cost: int | float | str


def _kernel(r: Reader, directory: Path, raw: object) -> _KernelTable:
    table = r.table(raw, ("kernel",))
    r.known(table, KERNEL_KEYS, ("kernel",))
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
            fixed[pname] = checked_value(probe, params[pname], value, ("params", pname))
        except ManifestError as error:
            raise fail(str(error).split(": ", 1)[-1]) from None
        # A default the template gives stays: a query that leaves the parameter
        # out means it, and an entry fixing another value refuses that query, as
        # the old exact-match supports did. Without one the fixed value stands
        # in (a shape the query never names, such as a token count).
        default = params[pname].default
        params[pname] = replace(
            params[pname],
            choices=(fixed[pname],),
            default=fixed[pname] if default is None else default,
        )
    return fixed


def _case_names(cases: object) -> bool:
    """Whether an entry's ``cases`` names cases once each, ``default`` among them."""
    return (
        isinstance(cases, list)
        and all(isinstance(c, str) for c in cases)
        and "default" in cases
        and len(set(cases)) == len(cases)
    )


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
    params = {
        name: read_param(r, name, raw, family_name, BYTE_UNIT_KERNELS)
        for name, raw in raw_params.items()
    }
    entry = _overrides(toml, params, kernel.formats, overrides or {}, source)
    derived = read_derived(r, data.get("derived", {}), params)
    # A placed derived value (it reads base(x)) is known only once the regions
    # are: a shape, a pin or the cost cannot read it, only [abi].
    placed = placed_derived(r, derived, declared_tensors(r, data))
    scalars = (*params, *derived)
    if isinstance(kernel.cost, str):
        integers = [n for n, p in params.items() if p.type == "int"]
        check_expression(r, kernel.cost, (*integers, *derived), ("kernel", "cost"))
    bools = tuple(n for n, p in params.items() if p.type == "bool")
    tensors = read_tensors(r, data, scalars, kernel.formats, bools, directory)
    numbers = [n for n, p in params.items() if p.type in ("int", "float")]
    for tensor in tensors.values():
        if isinstance(tensor.fill, str):
            path = (tensor.section, tensor.name, "fill")
            check_expression(r, tensor.fill, (*numbers, *derived), path)
    check_no_placed_reads(r, placed, tensors, kernel.cost)
    abi, abi_offsets, abi_when = read_abi(
        r, data.get("abi", {}), params, {*scalars, *tensors}, tensors, placed
    )
    check_placed_conditions(r, placed, tensors, abi_when, abi)
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
        golden=read_golden(r, directory, data.get("golden", {}), kernel.formats, tensors),
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
    cases = {name: read_case(r, manifest, name, raw) for name, raw in raw_cases.items()}
    manifest = replace(manifest, cases=MappingProxyType(cases))
    if "benchmark" in data:
        manifest = replace(manifest, benchmark=read_benchmark(r, manifest, data["benchmark"]))
    return manifest
