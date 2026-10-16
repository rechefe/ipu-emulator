"""The one place a kernel's parameters are validated: :func:`plan_case` (E06-03).

``plan_case(manifest, params, fmt)`` turns a request into a :class:`CasePlan`
(complete parameters, every tensor shape in ints) or raises
:class:`PlanError`. Placement, packing, CR programming, assembly and the runner
take the plan and never check again. The registry asks "does this kernel take
these parameters?" with :func:`check_params`, the same checks without a format.

Checks, in order:

1. Format: ``fmt`` is one of ``[kernel] formats``, and of the case's
   ``formats`` when it lists them.
2. Parameters: no unknown names, every parameter without a default given,
   each value of its type and in its domain (``min``, ``max``, ``multiple_of``,
   ``choices``).
3. Shapes: every tensor dimension is a positive int. A dimension is an int, a
   parameter or derived name, or an expression in a small grammar: int
   literals, names, ``+ - * / // %``, unary minus, and the calls ``ceil``,
   ``min`` and ``max``. Arithmetic is exact (fractions); ``/`` must end up
   inside ``ceil`` or divide evenly. Nothing else is evaluated.
4. Element scalars: a ``float`` parameter bound in ``[abi]`` is a scalar in the
   active format, so it must encode in ``fmt`` exactly. If it does not, the
   error names the nearest representable values; nothing is rounded silently.
"""

from __future__ import annotations

import ast
import math
import struct
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Any

import numpy as np

from ipu_emu.errors import ConfigurationError
from ipu_emu.ipu_math import DType, fp8_bytes_to_fp32, fp32_to_fp8_bytes
from ipu_harness.manifest import TOML_NAME, KernelManifest, ParamDecl
from ipu_isa.formats import FORMATS, ElementFormat

Value = int | float | bool | str

_PY_TYPES: Mapping[str, tuple[type, ...]] = {
    "int": (int,),
    "float": (int, float),
    "bool": (bool,),
    "str": (str,),
}


class PlanError(ConfigurationError):
    """A request a kernel does not accept (setup time, Q59). Names the manifest and the key."""

    def __init__(self, manifest: KernelManifest, key: str, message: str) -> None:
        self.toml = manifest.directory / TOML_NAME
        self.key = key
        super().__init__(f"{self.toml}: {key}: {message}")


@dataclass(frozen=True)
class CasePlan:
    manifest: KernelManifest
    format: ElementFormat
    params: Mapping[str, Value]  # complete: defaults filled, ints for float params made floats
    shapes: Mapping[str, tuple[int, ...]]  # every tensor shape in ints
    derived: Mapping[str, int]  # [derived] values that are shape-grammar expressions


# --- shape expressions --------------------------------------------------------------------

_BINARY: Mapping[type, Callable[[Fraction, Fraction], Fraction]] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: Fraction(a // b),
    ast.Mod: lambda a, b: a % b,
}
_CALLS: Mapping[str, Callable[..., Fraction]] = {
    "ceil": lambda x: Fraction(math.ceil(x)),
    "min": min,
    "max": max,
}


class ExpressionError(ValueError):
    """A shape or derived expression outside the grammar, or one that does not evaluate."""


def _evaluate(node: ast.AST, names: Mapping[str, Fraction]) -> Fraction:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body, names)
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return Fraction(node.value)
    if isinstance(node, ast.Name):
        if node.id not in names:
            raise ExpressionError(f"{node.id!r} is no integer parameter or derived value")
        return names[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_evaluate(node.operand, names)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _evaluate(node.left, names), _evaluate(node.right, names)
        if right == 0 and isinstance(node.op, ast.Div | ast.FloorDiv | ast.Mod):
            raise ExpressionError("division by zero")
        return _BINARY[type(node.op)](left, right)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _CALLS:
        if node.keywords or not node.args:
            raise ExpressionError(f"{node.func.id}() takes positional arguments only")
        return _CALLS[node.func.id](*(_evaluate(a, names) for a in node.args))
    raise ExpressionError(f"{ast.dump(node)[:60]} is outside the shape grammar")


def evaluate(expression: str, names: Mapping[str, int]) -> int:
    """``expression`` (the shape grammar) over integer ``names``, as an exact int.

    Raises:
        ExpressionError: Outside the grammar, an unknown name, or not a whole number.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ExpressionError(f"{expression!r} does not parse: {error.msg}") from None
    value = _evaluate(tree, {k: Fraction(v) for k, v in names.items()})
    if value.denominator != 1:
        raise ExpressionError(f"{expression!r} is {value}, not a whole number")
    return int(value)


# --- element scalars ----------------------------------------------------------------------


def _fp32(value: float) -> float:
    return float(struct.unpack("<f", struct.pack("<f", value))[0])


def _fp8_values(fmt: ElementFormat) -> list[float]:
    decoded = fp8_bytes_to_fp32(bytes(range(256)), DType(FORMATS[fmt].exp_bits))
    return sorted({float(v) for v in decoded if np.isfinite(v)})


def representable(value: float, fmt: ElementFormat) -> tuple[bool, tuple[float, ...]]:
    """Whether ``value`` encodes exactly in ``fmt``; if not, its nearest neighbours there.

    The FP8 formats round-trip through the emulator's codec until E04-13 moves
    the element codec into ``ipu_isa.formats``.
    """
    info = FORMATS[fmt]
    if not info.is_float:
        lo, hi = -(1 << (info.element_bits - 1)), (1 << (info.element_bits - 1)) - 1
        if float(value).is_integer() and lo <= value <= hi:
            return True, ()
        near = {min(max(math.floor(value), lo), hi), min(max(math.ceil(value), lo), hi)}
        return False, tuple(float(v) for v in sorted(near))
    if fmt is ElementFormat.FP32:
        exact = math.isfinite(value) and _fp32(value) == value
        if exact:
            return True, ()
        below = float(np.nextafter(np.float32(value), np.float32(-np.inf)))
        above = float(np.nextafter(np.float32(value), np.float32(np.inf)))
        return False, (below, above)
    encoded = fp32_to_fp8_bytes(np.array([value], dtype=np.float32), DType(info.exp_bits))
    if float(fp8_bytes_to_fp32(encoded, DType(info.exp_bits))[0]) == value:
        return True, ()
    values = _fp8_values(fmt)
    below = [v for v in values if v < value]
    above = [v for v in values if v > value]
    return False, tuple(
        v for v in (below[-1] if below else None, above[0] if above else None) if v is not None
    )


# --- the checks ---------------------------------------------------------------------------


def _check_format(manifest: KernelManifest, fmt: ElementFormat, case: str | None) -> None:
    if fmt not in manifest.formats:
        allowed = ", ".join(f.value for f in manifest.formats)
        raise PlanError(
            manifest, "formats", f"format {fmt.value} is not one of the kernel's: {allowed}"
        )
    if case is None:
        return
    if case not in manifest.cases:
        raise PlanError(manifest, f"cases.{case}", f"no case {case!r}")
    listed = manifest.cases[case].formats
    if listed is not None and fmt not in listed:
        raise PlanError(
            manifest, f"cases.{case}.formats", f"format {fmt.value} is not one of the case's"
        )


def _check_value(manifest: KernelManifest, decl: ParamDecl, value: object) -> Value:
    key = f"params.{decl.name}"
    if not isinstance(value, _PY_TYPES[decl.type]) or (
        isinstance(value, bool) and decl.type != "bool"
    ):
        raise PlanError(manifest, key, f"{decl.name} = {value!r} is not {decl.type}")
    checked: Any = float(value) if decl.type == "float" else value  # type: ignore[arg-type]
    if decl.min is not None and checked < decl.min:
        raise PlanError(manifest, key, f"{decl.name} = {value!r} is below its min {decl.min}")
    if decl.max is not None and checked > decl.max:
        raise PlanError(manifest, key, f"{decl.name} = {value!r} is above its max {decl.max}")
    if decl.multiple_of and checked % decl.multiple_of:
        raise PlanError(
            manifest, key, f"{decl.name} = {value!r} is not a multiple of {decl.multiple_of}"
        )
    if decl.choices is not None and checked not in decl.choices:
        raise PlanError(
            manifest, key, f"{decl.name} = {value!r} is not one of {list(decl.choices)}"
        )
    result: Value = checked
    return result


def _check_params(manifest: KernelManifest, params: Mapping[str, object]) -> dict[str, Value]:
    unknown = sorted(set(params) - set(manifest.params))
    if unknown:
        raise PlanError(
            manifest,
            f"params.{unknown[0]}",
            f"{unknown[0]} = {params[unknown[0]]!r} is no parameter",
        )
    out: dict[str, Value] = {}
    for name, decl in manifest.params.items():
        if name in params:
            out[name] = _check_value(manifest, decl, params[name])
        elif decl.default is not None:
            default: Any = decl.default  # checked against the domain by load_manifest
            out[name] = default
        else:
            raise PlanError(
                manifest, f"params.{name}", f"{name} has no default and no value was given"
            )
    return out


def _names(manifest: KernelManifest, params: Mapping[str, Value]) -> dict[str, int]:
    """Integer parameters, then the [derived] values the shape grammar can evaluate."""
    names = {k: v for k, v in params.items() if isinstance(v, int) and not isinstance(v, bool)}
    for name, expression in manifest.derived.items():
        try:
            names[name] = evaluate(expression, names)
        except ExpressionError:
            continue  # a derived value outside the shape grammar is E06-06's (module:fn)
    return names


def _shapes(manifest: KernelManifest, names: Mapping[str, int]) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for tensor in manifest.tensors.values():
        dims = []
        for dim in tensor.shape:
            key = (
                f"{tensor.role}s.{tensor.name}.shape"
                if tensor.role != "scratch"
                else f"scratch.{tensor.name}.shape"
            )
            try:
                size = dim if isinstance(dim, int) else evaluate(dim, names)
            except ExpressionError as error:
                raise PlanError(manifest, key, f"{dim!r}: {error}") from None
            if size <= 0:
                raise PlanError(manifest, key, f"{dim!r} = {size}, not a positive size")
            dims.append(size)
        shapes[tensor.name] = tuple(dims)
    return shapes


def _check_scalars(
    manifest: KernelManifest, params: Mapping[str, Value], fmt: ElementFormat
) -> None:
    for name in manifest.abi:
        decl = manifest.params.get(name)
        if decl is None or decl.type != "float":
            continue
        raw: Any = params[name]
        value = float(raw)
        exact, near = representable(value, fmt)
        if not exact:
            nearest = " and ".join(repr(v) for v in near) or "none"
            raise PlanError(
                manifest,
                f"params.{name}",
                f"{name} = {value!r} does not encode exactly in {fmt.value}; nearest: {nearest}",
            )


def check_params(manifest: KernelManifest, params: Mapping[str, object]) -> None:
    """Checks 2 and 3 (no format): raises :class:`PlanError` when one fails.

    For the registry's "does this kernel accept these parameters?" (E06-16).
    """
    checked = _check_params(manifest, params)
    _shapes(manifest, _names(manifest, checked))


def plan_case(
    manifest: KernelManifest,
    params: Mapping[str, object],
    fmt: ElementFormat,
    *,
    case: str | None = None,
) -> CasePlan:
    """The only place a kernel's parameters are validated.

    Args:
        manifest: The kernel.
        params: Parameter values; missing ones take their defaults.
        fmt: The element format to run in.
        case: The case the values come from, if any (its ``formats`` limit applies).

    Returns:
        The plan every later stage takes.

    Raises:
        PlanError: The first check that fails, naming the key and the value.
    """
    _check_format(manifest, fmt, case)
    checked = _check_params(manifest, params)
    names = _names(manifest, checked)
    shapes = _shapes(manifest, names)
    _check_scalars(manifest, checked, fmt)
    derived = {k: v for k, v in names.items() if k in manifest.derived}
    return CasePlan(
        manifest=manifest,
        format=fmt,
        params=MappingProxyType(checked),
        shapes=MappingProxyType(shapes),
        derived=MappingProxyType(derived),
    )
