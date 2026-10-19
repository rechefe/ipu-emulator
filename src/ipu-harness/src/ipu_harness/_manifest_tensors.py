"""The ``[inputs]``, ``[outputs]`` and ``[scratch]`` tables of a ``kernel.toml``,
and the placed derived values that read a region's base (for
:mod:`ipu_harness.manifest`)."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ipu_harness._codec import limits
from ipu_harness._manifest_decls import ROLES, TENSOR_KEYS, TensorDecl
from ipu_harness._manifest_params import (
    check_expression,
    check_name,
    expression_names,
    read_condition,
)
from ipu_harness._toml_source import Reader, where
from ipu_harness.expressions import ROW_BYTES_NAME, base_tensor, placed_reads
from ipu_isa.formats import FORMATS, ElementFormat


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
            check_expression(r, dim, names, path)
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
            check_expression(r, row, (*names, ROW_BYTES_NAME), (*path, "row"))
        elif row < 0:
            raise r.fail((*path, "row"), f"{where((*path, 'row'))} {row} must not be negative")
        decl = replace(decl, row=row)
    if "after" in table:
        if "row" in table:
            raise r.fail((*path, "after"), f"{where(path)} is pinned by row; it takes no 'after'")
        decl = replace(decl, after=r.text(table["after"], (*path, "after")))
    return decl


def _tensor(
    r: Reader,
    section: str,
    name: str,
    raw: object,
    names: Iterable[str],
    formats: tuple[ElementFormat, ...],
) -> TensorDecl:
    role = ROLES[section]
    path = (section, name)
    check_name(r, path)
    table = r.table(raw, path)
    r.known(table, TENSOR_KEYS[role], path)
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


def read_tensors(
    r: Reader,
    data: Mapping[str, Any],
    scalars: tuple[str, ...],
    formats: tuple[ElementFormat, ...],
    bools: tuple[str, ...],
    directory: Path,
) -> dict[str, TensorDecl]:
    tensors: dict[str, TensorDecl] = {}
    for section in ROLES:
        for name, raw in r.table(data.get(section, {}), (section,)).items():
            if name in tensors or name in scalars:
                raise r.fail((section, name), f"{name!r} is declared twice")
            decl = _tensor(r, section, name, raw, scalars, formats)
            if isinstance(raw, dict) and "when" in raw:
                when = read_condition(r, raw["when"], bools, (section, name, "when"))
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


def declared_tensors(r: Reader, data: Mapping[str, Any]) -> list[str]:
    """Every tensor name the file declares, before its tables are read."""
    return [name for section in ROLES for name in r.table(data.get(section, {}), (section,))]


def placed_derived(
    r: Reader, derived: Mapping[str, str], tensors: Iterable[str]
) -> Mapping[str, frozenset[str]]:
    """The derived values that read a region's base (``placed_reads``), once
    each ``base(x)`` is checked to name a tensor."""
    declared = set(tensors)
    for name, expression in derived.items():
        path = ("derived", name)
        for read in expression_names(r, expression, path):
            tensor = base_tensor(read)
            if tensor is not None and tensor not in declared:
                raise r.fail(path, f"{where(path)} {expression!r}: {read} names no tensor")
    return placed_reads(derived)


def check_no_placed_reads(
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
        read = sorted(late & set(expression_names(r, text, path)))
        if read:
            raise r.fail(
                path,
                f"{where(path)} {text!r} reads {read}, which read a region's base or row_bytes: "
                "only [abi] may read such a value",
            )
