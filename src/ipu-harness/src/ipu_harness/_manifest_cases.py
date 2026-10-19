"""The ``[cases]`` and ``[benchmark]`` tables of a ``kernel.toml`` (for
:mod:`ipu_harness.manifest`)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ipu_harness._manifest_decls import (
    BENCHMARK_KEYS,
    CASE_KEYS,
    BenchmarkDecl,
    CaseDecl,
    KernelManifest,
)
from ipu_harness._manifest_params import checked_value
from ipu_harness._toml_source import Reader, where


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


def _expected(
    r: Reader, directory: Path, raw: object, path: tuple[str, ...], outputs: Iterable[str]
) -> Mapping[str, Path | str]:
    """A case's ``expected``: per output a ``.npy`` file, or a ``"file.py:fn"``
    (or ``"../file.py:fn"``, the family's) whose ``fn(plan)`` returns the values."""
    table = r.table(raw, path)
    valid = tuple(outputs)
    out: dict[str, Path | str] = {}
    for tensor, value in table.items():
        if tensor not in valid:
            raise r.fail(
                (*path, tensor), f"{where(path)} {tensor!r} is not one of the tensors {list(valid)}"
            )
        text = r.typed(value, str, (*path, tensor))
        if ".py:" in text:
            out[tensor] = r.shared_function(directory, text, (*path, tensor))
        else:
            out[tensor] = r.inside(directory, text, (*path, tensor))
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
        checked = [checked_value(r, m.params[key], v, (*path, key)) for v in items]
        values[key] = tuple(checked) if isinstance(value, list) else checked[0]
    values.update(m.fixed)
    return values


def read_case(r: Reader, m: KernelManifest, name: str, raw: object) -> CaseDecl:
    path = ("cases", name)
    table = r.table(raw, path)
    r.known(table, (*m.params, *CASE_KEYS), path)
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
        expected = _expected(r, m.directory, table["expected"], (*path, "expected"), outputs)
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


def read_benchmark(r: Reader, m: KernelManifest, raw: object) -> BenchmarkDecl:
    path = ("benchmark",)
    table = r.table(raw, path)
    r.known(table, BENCHMARK_KEYS, path)
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
