"""A kernel's cases, and the input tensors each runs on (E06-08, Q82).

:func:`expand` turns ``[cases]`` into :class:`Case` objects: a list-valued key
is a sweep axis, and a case with axes becomes one case per point of their
cartesian product, named ``<case>[k=v,...]``. :func:`inputs_for` gives every
input tensor of a planned case, from the first source that has it:

1. an override (``--input x=path.npy``);
2. the case's ``inputs`` file;
3. the case's ``generator`` (``"gen.py:make"``: ``make(plan, rng)`` returns
   logical tensors by name, for inputs that need structure);
4. the default generator: uniform values in ``[-1, 1)`` for a float format and
   over the whole INT8 range for INT8, or the input's ``range``.

Generated values are rounded to the format, so the reference sees exactly what
XMEM holds; a file's values must already be exact in the format (nothing is
rounded silently). Every stream comes from :func:`rng`, seeded by name, so a
tensor's values are the same in every process and independent of every other
tensor's.
"""

from __future__ import annotations

import importlib.util
import itertools
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import numpy.typing as npt

from ipu_emu.errors import ConfigurationError

# The element codec moves to ipu_isa.formats in E04-13 (as in params.py).
from ipu_emu.ipu_math import DType, fp8_bytes_to_fp32, fp32_to_fp8_bytes
from ipu_harness.manifest import CaseDecl, KernelManifest, Tier
from ipu_harness.params import CasePlan, PlanError
from ipu_isa.formats import FORMATS, ElementFormat

__all__ = ["Case", "expand", "inputs_for", "rng"]

Array = npt.NDArray[Any]


@dataclass(frozen=True)
class Case:
    """One case, sweeps expanded.

    Attributes:
        name: ``"default"``, or ``"sweep[rows=7]"`` for a point of a sweep.
        values: Parameter values, scalars only; a parameter the case does not
            give takes its default (a parameter with none is left to
            ``plan_case`` to report).
        formats: The formats it runs in: its own limit, else the kernel's.
        inputs: User input files, by tensor name.
        expected: User expected-output files, by tensor name.
        generator: ``"file.py:fn"`` that makes inputs, if any.
        tier: ``default`` or ``nightly`` (Q73).
        decl: The ``[cases.<name>]`` it comes from, for plan_case's ``case``.
    """

    name: str
    values: Mapping[str, object]
    formats: tuple[ElementFormat, ...]
    inputs: Mapping[str, Path]
    expected: Mapping[str, Path]
    generator: str | None
    tier: Tier
    decl: CaseDecl


def _spelled(value: object) -> str:
    """A sweep value as a case name shows it (TOML's spelling)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _points(decl: CaseDecl) -> list[tuple[str, dict[str, object]]]:
    """Each point of the case's sweep axes, in declaration order, as (suffix, values)."""
    axes = {k: v for k, v in decl.values.items() if isinstance(v, tuple)}
    if not axes:
        return [("", dict(decl.values))]
    points = []
    for combo in itertools.product(*axes.values()):
        point = dict(zip(axes, combo, strict=True))
        suffix = ",".join(f"{k}={_spelled(v)}" for k, v in point.items())
        points.append((f"[{suffix}]", {**decl.values, **point}))
    return points


def expand(manifest: KernelManifest) -> tuple[Case, ...]:
    """Every case of ``manifest``, in declaration order, sweeps expanded.

    Args:
        manifest: The kernel.

    Returns:
        The cases; a sweep's points in the order of its axes' cartesian product.
    """
    defaults = {n: p.default for n, p in manifest.params.items() if p.default is not None}
    cases = []
    for decl in manifest.cases.values():
        for suffix, values in _points(decl):
            cases.append(
                Case(
                    name=decl.name + suffix,
                    values=MappingProxyType({**defaults, **values}),
                    formats=tuple(
                        f for f in decl.formats or manifest.formats if f in manifest.formats
                    ),
                    inputs=decl.inputs,
                    expected=decl.expected,
                    generator=decl.generator,
                    tier=decl.tier,
                    decl=decl,
                )
            )
    return tuple(cases)


def rng(kernel: str, case: str, tensor: str) -> np.random.Generator:
    """The random stream for one tensor of one case: seeded by the three names
    (CRC-32 of ``"kernel/case/tensor"``), so the same in every process."""
    return np.random.default_rng(
        np.random.SeedSequence(zlib.crc32(f"{kernel}/{case}/{tensor}".encode()))
    )


# --- the format ---------------------------------------------------------------------------


def _in_format(values: Array, fmt: ElementFormat) -> Array:
    """``values`` as the format holds them: int8, or float32 rounded to the format."""
    info = FORMATS[fmt]
    if not info.is_float:
        return np.clip(np.rint(values), -128, 127).astype(np.int8)
    as_fp32 = np.asarray(values, dtype=np.float32)
    if fmt is ElementFormat.FP32:
        return as_fp32
    dtype = DType(info.exp_bits)
    flat: Array = fp8_bytes_to_fp32(fp32_to_fp8_bytes(as_fp32.ravel(), dtype), dtype)
    return np.reshape(flat, as_fp32.shape)


def _exact(values: Array, fmt: ElementFormat) -> Array | None:
    """``values`` in the format if it holds them exactly, else None."""
    held = _in_format(values, fmt)
    return held if np.array_equal(held.astype(np.float64), values.astype(np.float64)) else None


def _generated(plan: CasePlan, case: Case, name: str) -> Array:
    """The default generator's values for input ``name``."""
    stream = rng(plan.manifest.name, case.name, name)
    shape = plan.shapes[name]
    info = FORMATS[plan.format]
    bounds = plan.manifest.tensors[name].value_range.get(plan.format)
    if info.is_float:
        lo, hi = bounds or (-1.0, 1.0)
        return _in_format(stream.uniform(lo, hi, size=shape), plan.format)
    lo, hi = bounds or (-128, 127)
    lo, hi = max(-128, int(np.ceil(lo))), min(127, int(np.floor(hi)))
    return stream.integers(lo, hi, size=shape, endpoint=True).astype(np.int8)


# --- the sources --------------------------------------------------------------------------


def _array(plan: CasePlan, name: str, value: object, where: str) -> Array:
    """``value`` checked as input ``name``: a numeric array of the planned shape."""
    if not isinstance(value, np.ndarray):
        raise ConfigurationError(f"{where}: {name} is {type(value).__name__}, not an array")
    if not (np.issubdtype(value.dtype, np.number) or value.dtype == np.bool_):
        raise ConfigurationError(f"{where}: {name} has dtype {value.dtype}, not a numeric one")
    shape = plan.shapes[name]
    if value.shape != shape:
        raise ConfigurationError(f"{where}: {name} has shape {value.shape}, the case's is {shape}")
    return value


def _loaded(plan: CasePlan, name: str, path: Path, where: str) -> Array:
    """Input ``name`` from a ``.npy`` file, exact in the format."""
    try:
        value = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ConfigurationError(f"{where}: cannot read {path} as {name}: {error}") from None
    array = _array(plan, name, value, f"{where} ({path})")
    held = _exact(array, plan.format)
    if held is None:
        raise ConfigurationError(
            f"{where} ({path}): {name} holds values {plan.format.value} cannot hold exactly"
        )
    return held


def _from_generator(plan: CasePlan, case: Case) -> dict[str, Array]:
    """What the case's ``generator`` makes, checked, in the format."""
    assert case.generator is not None
    key = f"cases.{case.decl.name}.generator"
    file, _, function = case.generator.partition(":")
    path = plan.manifest.directory / file
    spec = importlib.util.spec_from_file_location(f"_ipu_case_generator_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise PlanError(plan.manifest, key, f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    make = getattr(module, function, None)
    if not callable(make):
        raise PlanError(plan.manifest, key, f"{case.generator}: {file} has no function {function}")
    made = make(plan, rng(plan.manifest.name, case.name, "generator"))
    if not isinstance(made, Mapping):
        raise PlanError(
            plan.manifest, key, f"{case.generator} returned {type(made).__name__}, not a dict"
        )
    inputs = [t.name for t in plan.manifest.tensors.values() if t.role == "input"]
    out = {}
    for name, value in made.items():
        if name not in inputs:
            raise PlanError(
                plan.manifest, key, f"{case.generator} made {name!r}, which is no input"
            )
        try:
            array = _array(plan, name, value, case.generator)
        except ConfigurationError as error:
            raise PlanError(plan.manifest, key, str(error)) from None
        out[name] = _in_format(array.astype(np.float64), plan.format)
    return out


def inputs_for(
    case: Case, plan: CasePlan, *, overrides: Mapping[str, Path] | None = None
) -> dict[str, Array]:
    """Every input tensor of ``case`` planned as ``plan``.

    Args:
        case: The case.
        plan: Its plan (``plan_case``), for the format and the shapes.
        overrides: ``--input`` files, by tensor name; they win over the case.

    Returns:
        Each input, by name, in declaration order: int8 for INT8, else float32
        holding values exact in the format.

    Raises:
        ConfigurationError: An override names no input, or a file or the
            generator gives a tensor that is unreadable, of the wrong shape or
            dtype, or (a file) not exact in the format. Errors from the case's
            own files and generator are ``PlanError`` s located in ``kernel.toml``.
    """
    overrides = overrides or {}
    inputs = [t.name for t in plan.manifest.tensors.values() if t.role == "input"]
    unknown = sorted(set(overrides) - set(inputs))
    if unknown:
        raise ConfigurationError(
            f"--input {unknown[0]}=...: {unknown[0]!r} is no input of the kernel"
        )
    generated = _from_generator(plan, case) if case.generator else {}
    out = {}
    for name in inputs:
        if name in overrides:
            out[name] = _loaded(plan, name, overrides[name], f"--input {name}")
        elif name in case.inputs:
            key = f"cases.{case.decl.name}.inputs.{name}"
            try:
                out[name] = _loaded(plan, name, case.inputs[name], key)
            except ConfigurationError as error:
                raise PlanError(plan.manifest, key, str(error).split(": ", 1)[-1]) from None
        elif name in generated:
            out[name] = generated[name]
        else:
            out[name] = _generated(plan, case, name)
    return out
