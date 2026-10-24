"""Names, expressions, parameters, derived values and conditions of a
``kernel.toml`` (for :mod:`ipu_harness.manifest`)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

from ipu_harness._manifest_decls import CASE_KEYS, PARAM_KEYS, ParamDecl
from ipu_harness._toml_source import Reader, where
from ipu_harness.conditions import Condition
from ipu_harness.expressions import VARIANT_NAMES, ExpressionError, base_tensor, names_in

_PY_TYPES: Mapping[str, tuple[type, ...]] = {
    "int": (int,),
    "float": (int, float),
    "bool": (bool,),
    "str": (str,),
}

#: Names a template or a case reserves (E06-07: cr, lr, format, shapes).
_RESERVED_NAMES = frozenset({"cr", "lr", "format", "shapes", *VARIANT_NAMES, *CASE_KEYS})


def check_name(r: Reader, path: tuple[str, ...]) -> None:
    name = path[-1]
    if not name.isidentifier() or name.startswith("_") or name in _RESERVED_NAMES:
        raise r.fail(
            path,
            f"{where(path)}: {name!r} must be an identifier, not start with _, "
            "and not be cr, lr, format, shapes, row_bytes or a reserved case key",
        )


def check_expression(r: Reader, text: str, known: Iterable[str], path: tuple[str, ...]) -> None:
    """``text`` is in the shape grammar (E06-03) and names only ``known`` values."""
    try:
        names = names_in(text)
    except ExpressionError as error:
        raise r.fail(path, f"{where(path)}: {error}") from None
    unknown = sorted(set(names) - set(known))
    if unknown:
        raise r.fail(path, f"{where(path)} {text!r} names no parameter or derived value: {unknown}")


def expression_names(r: Reader, text: str, path: tuple[str, ...]) -> list[str]:
    try:
        return names_in(text)
    except ExpressionError as error:
        raise r.fail(path, f"{where(path)}: {error}") from None


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


def read_param(
    r: Reader, name: str, raw: object, kernel: str, byte_unit_kernels: frozenset[str]
) -> ParamDecl:
    """``[params.<name>]``; ``unit = "bytes"`` only for a kernel in ``byte_unit_kernels``."""
    path = ("params", name)
    check_name(r, path)
    table = r.table(raw, path)
    r.known(table, PARAM_KEYS, path)
    kind = r.require(table, "type", str, path)
    if kind not in _PY_TYPES:
        raise r.fail(
            (*path, "type"), f"{where(path)} type {kind!r} must be one of {', '.join(_PY_TYPES)}"
        )
    decl = ParamDecl(name, kind, **_bounds(r, table, kind, path))
    unit = table.get("unit", "count")
    if unit not in ("count", "rows", "bytes"):
        raise r.fail((*path, "unit"), f"{where(path)} unit {unit!r} must be count, rows or bytes")
    if unit == "bytes" and kernel not in byte_unit_kernels:
        raise r.fail((*path, "unit"), f"{where(path)} counts bytes; a kernel counts rows (L16)")
    decl = replace(decl, unit=unit)
    if "query_required" in table:
        required = r.typed(table["query_required"], bool, (*path, "query_required"))
        decl = replace(decl, query_required=bool(required))
    if "choices" in table:
        listed = r.typed(table["choices"], list, (*path, "choices"))
        if not listed:
            raise r.fail((*path, "choices"), f"{where(path)} choices must not be empty")
        decl = replace(
            decl, choices=tuple(checked_value(r, decl, c, (*path, "choices")) for c in listed)
        )
    if "default" in table:
        decl = replace(decl, default=checked_value(r, decl, table["default"], (*path, "default")))
    return decl


def checked_value(r: Reader, decl: ParamDecl, raw: object, path: tuple[str, ...]) -> object:
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


def read_derived(r: Reader, raw: object, params: Mapping[str, ParamDecl]) -> dict[str, str]:
    derived: dict[str, str] = {}
    for name, expression in r.table(raw, ("derived",)).items():
        path = ("derived", name)
        check_name(r, path)
        if name in params:
            raise r.fail(path, f"[derived] {name} is also a parameter")
        derived[name] = r.text(expression, path)
        bases = [n for n in expression_names(r, derived[name], path) if base_tensor(n) is not None]
        earlier = [d for d in derived if d != name]  # not itself: it would never resolve
        check_expression(r, derived[name], (*params, *earlier, *bases, *VARIANT_NAMES), path)
    return derived


def read_condition(
    r: Reader, raw: object, bools: Iterable[str], path: tuple[str, ...]
) -> Condition:
    """``"<bool param>"``, ``"not <bool param>"``, or such terms joined by ``and`` and ``or``."""
    condition = Condition.parse(r.text(raw, path))
    known = set(bools)
    if any(param not in known for param in condition.params):
        raise r.fail(
            path,
            f"{where(path)} {condition!s:.60} must be bool parameters (each may be "
            "'not' one) joined by 'and' or 'or'",
        )
    return condition
