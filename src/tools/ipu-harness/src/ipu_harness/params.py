"""The one place a kernel's parameters are validated: :func:`plan_case` (E06-03).

``plan_case(manifest, params, fmt)`` turns a request into a :class:`CasePlan`
(complete parameters, every tensor shape in ints, every derived value) or
raises :class:`PlanError`. Placement, packing, CR programming, assembly and the
runner take the plan and never check again. The registry asks "does this kernel
take these parameters, and in which formats?" with :func:`check_params`, the
same checks.

Checks, in order:

1. Format: ``fmt`` is one of ``[kernel] formats``, and of the case's
   ``formats`` when it lists them.
2. Parameters: no unknown names, every parameter without a default given,
   each value of its type, finite, and in its domain (``min``, ``max``,
   ``multiple_of``, ``choices``).
3. Shapes and derived values: every ``[derived]`` value (but a placed one,
   which reads a region's base: ``abi.register_writes`` computes it) and every
   tensor dimension evaluates, in the shape grammar (:mod:`ipu_harness.expressions`),
   to a whole number; dimensions are positive. In :func:`plan_case` (not
   :func:`check_params`, which routing calls), also every expression ``row``
   pin, and every scalar ``fill`` expression, exactly (``CasePlan.fills``).
4. Element scalars: a ``float`` parameter bound in ``[abi]`` is a scalar in the
   active format, so it must encode in ``fmt`` exactly. If it does not, the
   error names the nearest representable values; nothing is rounded silently.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from ipu_emu.errors import ConfigurationError
from ipu_harness.conditions import narrowed
from ipu_harness.expressions import ROW_BYTES, ExpressionError, evaluate, evaluate_exact
from ipu_harness.manifest import TOML_NAME, CaseDecl, KernelManifest, ParamDecl
from ipu_isa.formats import FORMATS, ElementFormat, element_values

Value = int | float | bool | str

_PY_TYPES: Mapping[str, tuple[type, ...]] = {
    "int": (int,),
    "float": (int, float),
    "bool": (bool,),
    "str": (str,),
}
_FP32_MAX = float(np.finfo(np.float32).max)


class PlanError(ConfigurationError):
    """A request a kernel does not accept (setup time, Q59).

    Attributes:
        toml: The kernel's ``kernel.toml``.
        key: The key that failed, as ``section.name[.field]``.
        reason: What is wrong, without the location (a routing refusal, E06-16).
    """

    def __init__(self, manifest: KernelManifest, key: str, message: str) -> None:
        self.toml = manifest.directory / TOML_NAME
        self.key = key
        self.reason = message
        super().__init__(f"{self.toml}: {key}: {message}")


@dataclass(frozen=True, eq=False)
class CasePlan:
    """A checked request: what every later stage takes (compared by identity).

    Attributes:
        manifest: The kernel.
        format: The element format it runs in.
        params: Every parameter, defaults filled; a float parameter's value is a float.
        shapes: Every tensor's shape, in ints.
        derived: Every ``[derived]`` value but the placed ones (they read a
            region's base: ``abi.register_writes`` computes them).
        pins: The first row of every pinned tensor (``row``), in ints.
        fills: Each scratch region's ``fill`` expression's value (a number
            ``fill`` is the tensor's own).
    """

    manifest: KernelManifest
    format: ElementFormat
    params: Mapping[str, Value]
    shapes: Mapping[str, tuple[int, ...]]
    derived: Mapping[str, int]
    pins: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))
    fills: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))


# --- element scalars ----------------------------------------------------------------------


@functools.cache
def _fp8_values(fmt: ElementFormat) -> tuple[float, ...]:
    # Every code's value (ipu_isa's element codec answers "does this encode exactly").
    return tuple(sorted({v for v in element_values(fmt) if math.isfinite(v)}))


def _neighbours(value: float, values: tuple[float, ...]) -> tuple[float, ...]:
    """The nearest of ``values`` below and above ``value``."""
    below = [v for v in values if v < value][-1:]
    above = [v for v in values if v > value][:1]
    return (*below, *above)


def _fp32_neighbours(value: float) -> tuple[float, ...]:
    """The float32 values on either side of ``value`` (one of them is its rounding)."""
    if abs(value) > _FP32_MAX:
        return (math.copysign(_FP32_MAX, value),)
    rounded = np.float32(value)
    if float(rounded) < value:
        return (float(rounded), float(np.nextafter(rounded, np.float32(np.inf))))
    return (float(np.nextafter(rounded, np.float32(-np.inf))), float(rounded))


def representable(value: float, fmt: ElementFormat) -> tuple[bool, tuple[float, ...]]:
    """Whether ``value`` encodes exactly in ``fmt``.

    Args:
        value: A finite number.
        fmt: The element format.

    Returns:
        ``(True, ())``, or ``(False, nearest)``: the representable values on
        either side of ``value`` (one, at the edge of the format's range).
    """
    if not math.isfinite(value):
        return False, ()
    info = FORMATS[fmt]
    if not info.is_float:
        lo, hi = -(1 << (info.element_bits - 1)), (1 << (info.element_bits - 1)) - 1
        if value < lo or value > hi:
            return False, (float(lo if value < lo else hi),)
        if value.is_integer():
            return True, ()
        return False, (float(math.floor(value)), float(math.ceil(value)))
    if fmt is ElementFormat.FP32:
        exact = abs(value) <= _FP32_MAX and float(np.float32(value)) == value
        return (True, ()) if exact else (False, _fp32_neighbours(value))
    values = _fp8_values(fmt)
    if value in values:
        return True, ()
    return False, _neighbours(value, values)


# --- the checks ---------------------------------------------------------------------------


def _check_format(manifest: KernelManifest, fmt: ElementFormat, case: CaseDecl | None) -> None:
    if fmt not in manifest.formats:
        allowed = ", ".join(f.value for f in manifest.formats)
        raise PlanError(
            manifest, "formats", f"format {fmt.value} is not one of the kernel's: {allowed}"
        )
    if case is not None and case.formats is not None and fmt not in case.formats:
        raise PlanError(
            manifest, f"cases.{case.name}.formats", f"format {fmt.value} is not one of the case's"
        )


def _check_number(manifest: KernelManifest, decl: ParamDecl, value: int | float) -> int | float:
    key = f"params.{decl.name}"
    if decl.type == "float":
        try:
            value = float(value)
        except OverflowError:
            raise PlanError(
                manifest, key, f"{decl.name} = {value!r} is too large for a float"
            ) from None
        if not math.isfinite(value):
            raise PlanError(manifest, key, f"{decl.name} = {value!r} is not finite")
    if decl.min is not None and value < decl.min:
        raise PlanError(manifest, key, f"{decl.name} = {value!r} is below its min {decl.min}")
    if decl.max is not None and value > decl.max:
        raise PlanError(manifest, key, f"{decl.name} = {value!r} is above its max {decl.max}")
    if decl.multiple_of and value % decl.multiple_of:
        raise PlanError(
            manifest, key, f"{decl.name} = {value!r} is not a multiple of {decl.multiple_of}"
        )
    return value


def _check_value(manifest: KernelManifest, decl: ParamDecl, value: object) -> Value:
    key = f"params.{decl.name}"
    if not isinstance(value, _PY_TYPES[decl.type]) or (
        isinstance(value, bool) and decl.type != "bool"
    ):
        raise PlanError(manifest, key, f"{decl.name} = {value!r} is not {decl.type}")
    checked: Value
    if isinstance(value, bool | str):
        checked = value
    elif isinstance(value, int | float):
        checked = _check_number(manifest, decl, value)
    else:  # pragma: no cover - the type check above admits nothing else
        raise PlanError(manifest, key, f"{decl.name} = {value!r} is not {decl.type}")
    if decl.choices is not None and checked not in decl.choices:
        raise PlanError(
            manifest, key, f"{decl.name} = {value!r} is not one of {list(decl.choices)}"
        )
    return checked


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
        elif decl.default is not None:  # TOML has no null: None means "no default"
            out[name] = _check_value(manifest, decl, decl.default)
        else:
            raise PlanError(
                manifest, f"params.{name}", f"{name} has no default and no value was given"
            )
    return out


def _derived(manifest: KernelManifest, params: Mapping[str, Value]) -> dict[str, int]:
    """Every [derived] value but the placed ones, in declaration order."""
    names = {k: v for k, v in params.items() if isinstance(v, int) and not isinstance(v, bool)}
    derived: dict[str, int] = {}
    for name, expression in manifest.derived.items():
        if name in manifest.placed:
            continue  # it reads a region's base: abi.register_writes computes it
        try:
            derived[name] = names[name] = evaluate(expression, names)
        except ExpressionError as error:
            raise PlanError(manifest, f"derived.{name}", f"{expression!r}: {error}") from None
    return derived


def _shapes(manifest: KernelManifest, names: Mapping[str, int]) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for tensor in manifest.tensors.values():
        key = f"{tensor.section}.{tensor.name}.shape"
        dims = []
        for dim in tensor.shape:
            try:
                size = dim if isinstance(dim, int) else evaluate(dim, names)
            except ExpressionError as error:
                raise PlanError(manifest, key, f"{dim!r}: {error}") from None
            if size <= 0:
                raise PlanError(manifest, key, f"{dim!r} = {size}, not a positive size")
            dims.append(size)
        shapes[tensor.name] = tuple(dims)
    return shapes


def _fills(
    manifest: KernelManifest, params: Mapping[str, Value], derived: Mapping[str, int]
) -> dict[str, float]:
    """Each scratch region's ``fill`` expression, evaluated exactly over the
    int and float parameters and the derived values (``"-1 / channels"``)."""
    numbers = {
        k: v for k, v in params.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    fills: dict[str, float] = {}
    for tensor in manifest.tensors.values():
        if isinstance(tensor.fill, str):
            try:
                fills[tensor.name] = float(evaluate_exact(tensor.fill, {**numbers, **derived}))
            except (ExpressionError, OverflowError) as error:  # too large for a float
                key = f"{tensor.section}.{tensor.name}.fill"
                raise PlanError(manifest, key, f"{tensor.fill!r}: {error}") from None
    return fills


def _pins(manifest: KernelManifest, names: Mapping[str, int]) -> dict[str, int]:
    pins: dict[str, int] = {}
    for tensor in manifest.tensors.values():
        if tensor.row is None:
            continue
        key = f"{tensor.section}.{tensor.name}.row"
        try:
            row = tensor.row if isinstance(tensor.row, int) else evaluate(tensor.row, names)
        except ExpressionError as error:
            raise PlanError(manifest, key, f"{tensor.row!r}: {error}") from None
        if row < 0:
            raise PlanError(manifest, key, f"{tensor.row!r} = {row}, not a row")
        pins[tensor.name] = row
    return pins


def _check_scalars(
    manifest: KernelManifest, params: Mapping[str, Value], fmt: ElementFormat
) -> None:
    for name in manifest.abi:
        decl = manifest.params.get(name)
        value = params.get(name)
        if decl is None or decl.type != "float" or not isinstance(value, float):
            continue
        exact, near = representable(value, fmt)
        if not exact:
            nearest = " and ".join(repr(v) for v in near) or "none"
            raise PlanError(
                manifest,
                f"params.{name}",
                f"{name} = {value!r} does not encode exactly in {fmt.value}; nearest: {nearest}",
            )


@dataclass(frozen=True)
class ParamCheck:
    """What :func:`check_params` found.

    Attributes:
        params: Every parameter, defaults filled.
        formats: The kernel's formats whose element scalars accept the values.
        derived: Every ``[derived]`` value but the placed ones (for a routing
            cost, E06-16).
    """

    params: Mapping[str, Value]
    formats: tuple[ElementFormat, ...]
    derived: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))


def check_params(manifest: KernelManifest, params: Mapping[str, object]) -> ParamCheck:
    """Checks 2 and 3, and check 4 per format: for the registry (E06-16).

    Args:
        manifest: The kernel.
        params: Parameter values; missing ones take their defaults.

    Returns:
        The complete parameters and the formats that accept them (possibly none).

    Raises:
        PlanError: Check 2 or 3 fails.
    """
    checked = _check_params(manifest, params)
    derived = _derived(manifest, checked)
    names = {k: v for k, v in checked.items() if isinstance(v, int) and not isinstance(v, bool)}
    _shapes(manifest, {**names, **derived})
    formats = []
    for fmt in manifest.formats:
        try:
            _check_scalars(manifest, checked, fmt)
        except PlanError:
            continue
        formats.append(fmt)
    return ParamCheck(MappingProxyType(checked), tuple(formats), MappingProxyType(derived))


def plan_case(
    manifest: KernelManifest,
    params: Mapping[str, object],
    fmt: ElementFormat,
    *,
    case: CaseDecl | None = None,
) -> CasePlan:
    """The only place a kernel's parameters are validated.

    Args:
        manifest: The kernel.
        params: Parameter values; missing ones take their defaults.
        fmt: The element format to run in.
        case: The case the values come from, if any (its ``formats`` limit
            applies; a sweep's expanded cases pass their base case).

    Returns:
        The plan every later stage takes.

    Raises:
        PlanError: The first check that fails, naming the key and the value.
    """
    _check_format(manifest, fmt, case)
    checked = _check_params(manifest, params)
    manifest = narrowed(manifest, checked)  # the case's tensors and bindings (``when``)
    derived = _derived(manifest, checked)
    names = {k: v for k, v in checked.items() if isinstance(v, int) and not isinstance(v, bool)}
    shapes = _shapes(manifest, {**names, **derived})
    pins = _pins(manifest, {**names, **derived, ROW_BYTES: FORMATS[fmt].row_bytes})
    fills = _fills(manifest, checked, derived)
    _check_scalars(manifest, checked, fmt)
    return CasePlan(
        manifest=manifest,
        format=fmt,
        params=MappingProxyType(checked),
        shapes=MappingProxyType(shapes),
        derived=MappingProxyType(derived),
        pins=MappingProxyType(pins),
        fills=MappingProxyType(fills),
    )
