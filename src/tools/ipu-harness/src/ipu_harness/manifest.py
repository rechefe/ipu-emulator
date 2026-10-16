"""``kernel.toml``: a kernel's whole declaration apart from its program (E06-02).

:func:`load_manifest` is the only reader of ``kernel.toml``. It checks the file
against the schema below and returns a frozen :class:`KernelManifest`; every
later stage reads that view, never the TOML. An error names the file, the line
and the key: ``kernels/elementwise/scale/kernel.toml:14: [abi] binds CR1, a
hardwired constant``.

Schema
------

``[kernel]``
    ``op`` (str, required): the operation, also the shared reference's name.
    ``formats`` (list, required): element formats the kernel runs in, by their
    ``ipu_isa.formats.ElementFormat`` names (``fp32``, ``int8``, ``e1``..``e7``,
    ``int32``). ``asm`` (path): the program, default ``<folder>.asm``.
    ``stages`` (``"file.py:fn"``): a multi-program pipeline (E06-14).

``[params.<name>]`` (or ``<name> = { ... }`` under ``[params]``)
    ``type`` (required): ``int``, ``float``, ``bool`` or ``str``. ``min``,
    ``max``, ``multiple_of`` (int only), ``choices`` (list), ``default``, and
    ``unit``: ``count`` (default), ``rows`` or ``bytes`` (``bytes`` only for a
    kernel in :data:`BYTE_UNIT_KERNELS`, L16).

``[inputs.<name>]``, ``[outputs.<name>]``, ``[scratch.<name>]``
    ``shape`` (list, required): ints, parameter names or simple expressions over
    them (E06-03). ``row``: pin the region's first XMEM row (Q81). ``pack``
    (inputs): a named pack transform (E06-05). ``store`` (outputs):
    ``element`` (default) or ``accumulator``. ``fill`` (scratch): the value the
    region is filled with, default 0. Scratch regions are placed and filled,
    never compared.

``[abi]``
    ``<name> = "CR2"`` .. ``"CR15"``, or ``"LR0"`` .. ``"LR15"`` (a preset, E06-06).
    ``CR0`` and ``CR1`` are hardwired constants and cannot be bound. Every key
    names a parameter, a tensor or a derived value, and every tensor is bound
    (its base row must reach the program).

``[derived]``
    ``<name> = "<expression over params>"`` (E06-06).

``[cases.<name>]``
    Parameter values: a scalar, or a list (a sweep axis). Reserved keys:
    ``formats`` (a subset of the kernel's), ``inputs`` and ``expected``
    (tensor name to a user ``.npy`` file, Q82), ``generator``
    (``"file.py:fn"``), ``tier`` (``default`` or ``nightly``, Q73).
    ``[cases.default]`` is required.

``[golden]``
    ``reference`` (``"file.py:fn"``; default ``reference.py:reference`` when the
    kernel has a ``reference.py``), ``op`` (a shared reference in
    ``ipu_kernels.reference``), ``tolerance`` (format to tier name, E06-10).

Every file path (``asm``, ``inputs``, ``expected``, ``reference``,
``generator``, ``stages``) stays inside the kernel directory: ``..`` and
absolute paths are refused. An unknown section or key is an error that names
the closest valid one.
"""

from __future__ import annotations

import difflib
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Literal, cast

from ipu_isa.errors import IpuError
from ipu_isa.formats import ElementFormat
from ipu_isa.registers import REGISTER_DEFINITIONS

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

TOML_NAME = "kernel.toml"

#: Kernels allowed ``unit = "bytes"`` parameters (L16). E07 lists the legacy
#: kernels it migrates that still count in bytes; a new kernel counts in rows.
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
)
_KERNEL_KEYS = ("op", "formats", "asm", "stages")
_PARAM_KEYS = ("type", "min", "max", "multiple_of", "choices", "default", "unit")
_TENSOR_KEYS: Mapping[Role, tuple[str, ...]] = {
    "input": ("shape", "row", "pack"),
    "output": ("shape", "row", "store"),
    "scratch": ("shape", "row", "fill"),
}
_CASE_KEYS = ("formats", "inputs", "expected", "generator", "tier")
_GOLDEN_KEYS = ("reference", "op", "tolerance")
_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")


class ManifestError(IpuError, ValueError):
    """A ``kernel.toml`` that breaks the schema; the message starts ``<file>:<line>:``."""


@dataclass(frozen=True)
class ParamDecl:
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
    name: str
    role: Role
    shape: tuple[int | str, ...]  # ints, or param names / simple expressions (E06-03)
    row: int | None = None  # Q81 pin
    pack: str | None = None  # inputs: a named transform (E06-05)
    store: Store = "element"  # outputs
    fill: int | float = 0  # scratch


@dataclass(frozen=True)
class CaseDecl:
    name: str
    values: Mapping[str, object]  # scalars or lists (lists are sweep axes)
    formats: tuple[ElementFormat, ...] | None = None
    inputs: Mapping[str, Path] = field(default_factory=lambda: MappingProxyType({}))
    expected: Mapping[str, Path] = field(default_factory=lambda: MappingProxyType({}))
    generator: str | None = None  # "file.py:fn"
    tier: Tier = "default"  # Q73


@dataclass(frozen=True)
class GoldenDecl:
    reference: str | None = None  # "file.py:fn"
    op: str | None = None  # a shared reference in ipu_kernels.reference
    tolerance: Mapping[ElementFormat, str] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True)
class KernelManifest:
    name: str  # the folder's name, or the registry.toml entry's (E06-17)
    directory: Path
    asm: Path
    op: str
    formats: tuple[ElementFormat, ...]
    params: Mapping[str, ParamDecl]
    tensors: Mapping[str, TensorDecl]
    abi: Mapping[str, str]  # name -> "CR2".."CR15" or "LR0".."LR15"
    derived: Mapping[str, str]
    cases: Mapping[str, CaseDecl]  # holds "default"
    golden: GoldenDecl
    stages: str | None = None  # "stages.py:run" (E06-14)


# --- where a key is in the file ---------------------------------------------------------


def _header(line: str) -> tuple[str, ...] | None:
    m = re.match(r"\s*\[([^\[\]]+)\]\s*(#.*)?$", line)
    if not m:
        return None
    return tuple(part.strip().strip("\"'") for part in m.group(1).split("."))


class _Locator:
    """Line numbers of tables and keys, read from the source text (TOML parsers keep none)."""

    def __init__(self, text: str) -> None:
        self.lines: dict[tuple[str, ...], int] = {}
        table: tuple[str, ...] = ()
        for number, line in enumerate(text.splitlines(), start=1):
            header = _header(line)
            if header is not None:
                table = header
                self.lines.setdefault(table, number)
                continue
            m = re.match(r"\s*([A-Za-z0-9_\-]+|\"[^\"]*\"|'[^']*')\s*=", line)
            if m:
                self.lines.setdefault((*table, m.group(1).strip("\"'")), number)

    def line(self, path: tuple[str, ...]) -> int:
        """The line of ``path``; else of the first key inside it; else of its nearest
        enclosing key or table; else 1."""
        if path in self.lines:
            return self.lines[path]
        inside = [n for p, n in self.lines.items() if p[: len(path)] == path]
        if path and inside:
            return min(inside)
        for end in range(len(path) - 1, 0, -1):
            if path[:end] in self.lines:
                return self.lines[path[:end]]
        return 1


@dataclass
class _Reader:
    """Raises :class:`ManifestError` at the line of a key path."""

    toml: Path
    locator: _Locator

    def fail(self, path: tuple[str, ...], message: str) -> ManifestError:
        return ManifestError(f"{self.toml}:{self.locator.line(path)}: {message}")

    def table(self, value: object, path: tuple[str, ...], what: str) -> Mapping[str, Any]:
        if not isinstance(value, dict):
            raise self.fail(path, f"{what} must be a table")
        return cast(Mapping[str, Any], value)

    def known(
        self, table: Mapping[str, Any], valid: Iterable[str], path: tuple[str, ...], what: str
    ) -> None:
        valid = tuple(valid)
        for key in table:
            if key not in valid:
                close = difflib.get_close_matches(key, valid, n=1)
                hint = f"; did you mean {close[0]!r}?" if close else f"; valid: {', '.join(valid)}"
                raise self.fail((*path, key), f"unknown key {key!r} in {what}{hint}")

    def require(
        self, table: Mapping[str, Any], key: str, kind: type, path: tuple[str, ...], what: str
    ) -> Any:
        if key not in table:
            raise self.fail(path, f"{what} needs {key!r}")
        return self.typed(table[key], kind, (*path, key), f"{what} {key}")

    def typed(
        self, value: object, kind: type | tuple[type, ...], path: tuple[str, ...], what: str
    ) -> Any:
        kinds = kind if isinstance(kind, tuple) else (kind,)
        if not isinstance(value, kinds) or (isinstance(value, bool) and bool not in kinds):
            names = " or ".join(k.__name__ for k in kinds)
            raise self.fail(path, f"{what} must be {names}, not {value!r}")
        return value

    def format(self, value: object, path: tuple[str, ...]) -> ElementFormat:
        try:
            return ElementFormat(value)
        except ValueError:
            valid = ", ".join(f.value for f in ElementFormat)
            raise self.fail(path, f"unknown element format {value!r}; valid: {valid}") from None

    def inside(self, directory: Path, value: str, path: tuple[str, ...], what: str) -> Path:
        file = PurePosixPath(value)
        if file.is_absolute() or ".." in file.parts or not value:
            raise self.fail(path, f"{what} {value!r} must be a path inside the kernel directory")
        return directory / file

    def function(self, directory: Path, value: object, path: tuple[str, ...], what: str) -> str:
        text = self.typed(value, str, path, what)
        file, sep, fn = text.partition(":")
        if not sep or not fn.isidentifier():
            raise self.fail(path, f"{what} {text!r} must be 'file.py:function'")
        self.inside(directory, file, path, what)
        return str(text)


# --- the sections --------------------------------------------------------------------------


def _param(r: _Reader, name: str, raw: object, kernel: str) -> ParamDecl:
    path = ("params", name)
    table = r.table(raw, path, f"[params] {name}")
    r.known(table, _PARAM_KEYS, path, f"[params] {name}")
    kind = r.require(table, "type", str, path, f"[params] {name}")
    if kind not in _PY_TYPES:
        raise r.fail(
            (*path, "type"), f"[params] {name} type {kind!r} must be one of {', '.join(_PY_TYPES)}"
        )
    pytypes = _PY_TYPES[kind]
    bounds = {
        k: r.typed(table[k], (int, float), (*path, k), f"[params] {name} {k}")
        for k in ("min", "max")
        if k in table
    }
    multiple_of = None
    if "multiple_of" in table:
        if kind != "int":
            raise r.fail((*path, "multiple_of"), f'[params] {name} multiple_of needs type = "int"')
        multiple_of = r.typed(
            table["multiple_of"], int, (*path, "multiple_of"), f"[params] {name} multiple_of"
        )
    choices = None
    if "choices" in table:
        listed = r.typed(table["choices"], list, (*path, "choices"), f"[params] {name} choices")
        choices = tuple(
            r.typed(c, pytypes, (*path, "choices"), f"[params] {name} choice") for c in listed
        )
    unit = table.get("unit", "count")
    if unit not in ("count", "rows", "bytes"):
        raise r.fail((*path, "unit"), f"[params] {name} unit {unit!r} must be count, rows or bytes")
    if unit == "bytes" and kernel not in BYTE_UNIT_KERNELS:
        raise r.fail((*path, "unit"), f"[params] {name} counts bytes; a kernel counts rows (L16)")
    decl = ParamDecl(
        name, kind, bounds.get("min"), bounds.get("max"), multiple_of, choices, None, unit
    )
    if "default" in table:
        decl = replace(decl, default=_value(r, decl, table["default"], (*path, "default")))
    return decl


def _value(r: _Reader, decl: ParamDecl, raw: object, path: tuple[str, ...]) -> object:
    """``raw`` checked against ``decl``'s type and domain."""
    value: Any = r.typed(raw, _PY_TYPES[decl.type], path, f"{decl.name}")
    out_of_range = (decl.min is not None and value < decl.min) or (
        decl.max is not None and value > decl.max
    )
    if out_of_range or (decl.multiple_of and value % decl.multiple_of):
        raise r.fail(path, f"{decl.name} = {value!r} is outside its domain")
    if decl.choices is not None and value not in decl.choices:
        raise r.fail(path, f"{decl.name} = {value!r} is not one of {list(decl.choices)}")
    return value


def _shape(
    r: _Reader, raw: object, path: tuple[str, ...], names: Iterable[str]
) -> tuple[int | str, ...]:
    dims = r.typed(raw, list, path, "shape")
    known = set(names)
    out: list[int | str] = []
    for dim in dims:
        dim = r.typed(dim, (int, str), path, "a shape dimension")
        if isinstance(dim, str):
            unknown = [n for n in _IDENTIFIER.findall(dim) if n not in known]
            if unknown or not dim.strip():
                raise r.fail(
                    path, f"shape dimension {dim!r} names no parameter or derived value: {unknown}"
                )
        elif dim <= 0:
            raise r.fail(path, f"shape dimension {dim} must be positive")
        out.append(dim)
    return tuple(out)


def _tensor(r: _Reader, section: str, name: str, raw: object, names: Iterable[str]) -> TensorDecl:
    role = _ROLES[section]
    path = (section, name)
    table = r.table(raw, path, f"[{section}.{name}]")
    r.known(table, _TENSOR_KEYS[role], path, f"[{section}.{name}]")
    if "shape" not in table:
        raise r.fail(path, f"[{section}.{name}] needs 'shape'")
    decl = TensorDecl(name, role, _shape(r, table["shape"], (*path, "shape"), names))
    if "row" in table:
        row = r.typed(table["row"], int, (*path, "row"), "row")
        if row < 0:
            raise r.fail((*path, "row"), f"row {row} must not be negative")
        decl = replace(decl, row=row)
    if "pack" in table:
        decl = replace(decl, pack=r.typed(table["pack"], str, (*path, "pack"), "pack"))
    if "store" in table:
        if table["store"] not in ("element", "accumulator"):
            raise r.fail(
                (*path, "store"), f"store {table['store']!r} must be element or accumulator"
            )
        decl = replace(decl, store=table["store"])
    if "fill" in table:
        decl = replace(decl, fill=r.typed(table["fill"], (int, float), (*path, "fill"), "fill"))
    return decl


def _register(r: _Reader, name: str, raw: object) -> str:
    text = r.typed(raw, str, ("abi", name), f"[abi] {name}").upper()
    m = re.fullmatch(r"(CR|LR)(\d+)", text)
    banks = {"CR": REGISTER_DEFINITIONS["cr"]["count"], "LR": REGISTER_DEFINITIONS["lr"]["count"]}
    if not m or int(m.group(2)) >= banks[m.group(1)]:
        raise r.fail(("abi", name), f"[abi] {name} = {raw!r} is no CR or LR register")
    if m.group(1) == "CR" and int(m.group(2)) < 2:
        raise r.fail(("abi", name), f"[abi] binds {text}, a hardwired constant")
    return f"{m.group(1)}{int(m.group(2))}"


def _abi(r: _Reader, raw: object, names: set[str], tensors: Iterable[str]) -> Mapping[str, str]:
    table = r.table(raw, ("abi",), "[abi]")
    abi: dict[str, str] = {}
    used: dict[str, str] = {}
    for name, value in table.items():
        if name not in names:
            raise r.fail(("abi", name), f"[abi] {name} names no parameter, tensor or derived value")
        register = _register(r, name, value)
        if register in used:
            raise r.fail(("abi", name), f"[abi] {name} and {used[register]} both bind {register}")
        used[register] = name
        abi[name] = register
    for tensor in tensors:
        if tensor not in abi:
            raise r.fail(
                ("abi",),
                f"[abi] does not bind tensor {tensor!r}; its base row must reach the program",
            )
    return MappingProxyType(abi)


def _files(
    r: _Reader, directory: Path, raw: object, path: tuple[str, ...], allowed: Iterable[str]
) -> Mapping[str, Path]:
    table = r.table(raw, path, f"{path[-1]}")
    valid = tuple(allowed)
    out = {}
    for tensor, value in table.items():
        if tensor not in valid:
            raise r.fail(
                (*path, tensor), f"{path[-1]} {tensor!r} is not one of the tensors {list(valid)}"
            )
        text = r.typed(value, str, (*path, tensor), f"{path[-1]} {tensor}")
        out[tensor] = r.inside(directory, text, (*path, tensor), f"{path[-1]} {tensor}")
    return MappingProxyType(out)


def _case_values(
    r: _Reader, m: KernelManifest, path: tuple[str, ...], table: Mapping[str, Any]
) -> dict[str, object]:
    """The case's parameter values: a scalar, or a tuple (a sweep axis)."""
    values: dict[str, object] = {}
    for key, value in table.items():
        if key in m.params:
            items = value if isinstance(value, list) else [value]
            if not items:
                raise r.fail((*path, key), f"[cases.{path[-1]}] {key} sweeps over no values")
            checked = [_value(r, m.params[key], v, (*path, key)) for v in items]
            values[key] = tuple(checked) if isinstance(value, list) else checked[0]
    return values


def _case(r: _Reader, m: KernelManifest, name: str, raw: object) -> CaseDecl:
    path = ("cases", name)
    table = r.table(raw, path, f"[cases.{name}]")
    r.known(table, (*m.params, *_CASE_KEYS), path, f"[cases.{name}]")
    values = _case_values(r, m, path, table)
    case = CaseDecl(name, MappingProxyType(values))
    if "formats" in table:
        listed = r.typed(table["formats"], list, (*path, "formats"), "formats")
        formats = tuple(r.format(f, (*path, "formats")) for f in listed)
        extra = [f.value for f in formats if f not in m.formats]
        if extra:
            raise r.fail((*path, "formats"), f"[cases.{name}] formats {extra} are not the kernel's")
        case = replace(case, formats=formats)
    if "inputs" in table:
        inputs = [t.name for t in m.tensors.values() if t.role == "input"]
        case = replace(
            case, inputs=_files(r, m.directory, table["inputs"], (*path, "inputs"), inputs)
        )
    if "expected" in table:
        outputs = [t.name for t in m.tensors.values() if t.role == "output"]
        case = replace(
            case, expected=_files(r, m.directory, table["expected"], (*path, "expected"), outputs)
        )
    if "generator" in table:
        case = replace(
            case,
            generator=r.function(
                m.directory, table["generator"], (*path, "generator"), "generator"
            ),
        )
    if "tier" in table:
        if table["tier"] not in ("default", "nightly"):
            raise r.fail((*path, "tier"), f"tier {table['tier']!r} must be default or nightly")
        case = replace(case, tier=table["tier"])
    return case


def _golden(
    r: _Reader, directory: Path, raw: object, formats: tuple[ElementFormat, ...]
) -> GoldenDecl:
    table = r.table(raw, ("golden",), "[golden]")
    r.known(table, _GOLDEN_KEYS, ("golden",), "[golden]")
    reference = None
    if "reference" in table:
        reference = r.function(directory, table["reference"], ("golden", "reference"), "reference")
    elif (directory / "reference.py").is_file():
        reference = "reference.py:reference"
    op = r.typed(table["op"], str, ("golden", "op"), "op") if "op" in table else None
    tolerance: dict[ElementFormat, str] = {}
    if "tolerance" in table:
        tiers = r.table(table["tolerance"], ("golden", "tolerance"), "tolerance")
        for fmt, tier in tiers.items():
            key = r.format(fmt, ("golden", "tolerance", fmt))
            if key not in formats:
                raise r.fail(
                    ("golden", "tolerance", fmt),
                    f"tolerance for {fmt}, a format the kernel does not run in",
                )
            tolerance[key] = r.typed(tier, str, ("golden", "tolerance", fmt), f"tolerance {fmt}")
    return GoldenDecl(reference, op, MappingProxyType(tolerance))


# --- the loader ---------------------------------------------------------------------------


def _kernel(
    r: _Reader, directory: Path, raw: object
) -> tuple[str, tuple[ElementFormat, ...], Path, str | None]:
    table = r.table(raw, ("kernel",), "[kernel]")
    r.known(table, _KERNEL_KEYS, ("kernel",), "[kernel]")
    op = r.require(table, "op", str, ("kernel",), "[kernel]")
    listed = r.require(table, "formats", list, ("kernel",), "[kernel]")
    formats = tuple(r.format(f, ("kernel", "formats")) for f in listed)
    if not formats or len(set(formats)) != len(formats):
        raise r.fail(
            ("kernel", "formats"), "[kernel] formats must list each format once, at least one"
        )
    asm_name = r.typed(table.get("asm", f"{directory.name}.asm"), str, ("kernel", "asm"), "asm")
    asm = r.inside(directory, asm_name, ("kernel", "asm"), "asm")
    if not asm.is_file():
        raise r.fail(("kernel", "asm"), f"the program {asm_name} is not in the kernel directory")
    stages = (
        r.function(directory, table["stages"], ("kernel", "stages"), "stages")
        if "stages" in table
        else None
    )
    return op, formats, asm, stages


def _sections(r: _Reader, data: Mapping[str, Any]) -> None:
    r.known(data, _SECTIONS, (), TOML_NAME)
    for section in ("kernel", "golden"):
        if section not in data:
            raise r.fail((), f"{TOML_NAME} needs a [{section}] section")


def _apply_overrides(
    r: _Reader, params: dict[str, ParamDecl], overrides: Mapping[str, Any]
) -> str | None:
    unknown = set(overrides) - {"name", "params"}
    if unknown:
        raise r.fail((), f"unknown overrides {sorted(unknown)}; valid: name, params")
    for name, value in overrides.get("params", {}).items():
        if name not in params:
            raise r.fail(("params",), f"override for {name!r}, which is no parameter")
        params[name] = replace(
            params[name], default=_value(r, params[name], value, ("params", name))
        )
    name = overrides.get("name")
    return str(name) if name is not None else None


def load_manifest(directory: Path, *, overrides: Mapping[str, Any] | None = None) -> KernelManifest:
    """Read and check ``directory/kernel.toml``; the frozen view of it.

    Args:
        directory: The kernel's folder (holding ``kernel.toml`` and its program).
        overrides: From a family ``registry.toml`` entry (E06-17): ``name`` and
            ``params`` (parameter name to its default).

    Returns:
        The manifest.

    Raises:
        ManifestError: The file breaks the schema; the message names its line.
    """
    toml = directory / TOML_NAME
    text = toml.read_text(encoding="utf-8")
    r = _Reader(toml, _Locator(text))
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise ManifestError(f"{toml}: {error}") from None
    _sections(r, data)
    op, formats, asm, stages = _kernel(r, directory, data["kernel"])
    raw_params = r.table(data.get("params", {}), ("params",), "[params]")
    params = {name: _param(r, name, raw, directory.name) for name, raw in raw_params.items()}
    name = _apply_overrides(r, params, overrides or {}) or directory.name
    raw_derived = r.table(data.get("derived", {}), ("derived",), "[derived]")
    derived = {k: r.typed(v, str, ("derived", k), f"[derived] {k}") for k, v in raw_derived.items()}
    clash = sorted(set(derived) & set(params))
    if clash:
        raise r.fail(("derived", clash[0]), f"[derived] {clash[0]} is also a parameter")
    scalars = (*params, *derived)
    tensors: dict[str, TensorDecl] = {}
    for section in _ROLES:
        for tname, raw in r.table(data.get(section, {}), (section,), f"[{section}]").items():
            if tname in tensors or tname in scalars:
                raise r.fail((section, tname), f"{tname!r} is declared twice")
            tensors[tname] = _tensor(r, section, tname, raw, scalars)
    abi = _abi(r, data.get("abi", {}), {*scalars, *tensors}, tensors)
    manifest = KernelManifest(
        name=name,
        directory=directory,
        asm=asm,
        op=op,
        formats=formats,
        params=MappingProxyType(params),
        tensors=MappingProxyType(tensors),
        abi=abi,
        derived=MappingProxyType(derived),
        cases=MappingProxyType({}),
        golden=_golden(r, directory, data["golden"], formats),
        stages=stages,
    )
    raw_cases = r.table(data.get("cases", {}), ("cases",), "[cases]")
    if "default" not in raw_cases:
        raise r.fail(("cases",), f"{TOML_NAME} needs [cases.default]")
    cases = {cname: _case(r, manifest, cname, raw) for cname, raw in raw_cases.items()}
    return replace(manifest, cases=MappingProxyType(cases))
