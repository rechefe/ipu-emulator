"""A kernel's cases, and the input tensors each runs on (E06-08, Q82).

:func:`expand` turns ``[cases]`` into :class:`Case` objects: a list-valued key
is a sweep axis, and a case with axes becomes one case per point of their
cartesian product, named ``<case>[k=v,...]``. :func:`inputs_for` gives every
input tensor of a planned case, from the first source that has it:

1. an override (``--input x=path.npy``);
2. the case's ``inputs`` file;
3. the case's ``generator`` (``"gen.py:make"``: ``make(plan, rng)`` returns
   logical tensors by name, for inputs that need structure; the file is the
   kernel folder's, else its family folder's, so a family's templates can
   share one);
4. the default generator: uniform values in ``[-1, 1]`` for a float format and
   over the whole range of an integer one, or the input's ``range``.

Every tensor is rounded to the format (to nearest, saturating), so the
reference sees exactly what XMEM holds. A file's or a generator's values that
rounding changed are counted (:attr:`CaseInputs.rounded`) for the report, as
E06-10 rules; a value that is not finite is an error. Every stream comes from
:func:`rng`, seeded by name, so a tensor's values are the same in every
process and independent of every other tensor's.
"""

from __future__ import annotations

import importlib.util
import itertools
import math
import zlib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

import numpy as np

from ipu_emu.errors import ConfigurationError
from ipu_harness._codec import Array, held, limits
from ipu_harness._plugins import kernel_file
from ipu_harness.manifest import CaseDecl, KernelManifest, Tier
from ipu_harness.params import CasePlan, PlanError
from ipu_isa.formats import FORMATS, ElementFormat

__all__ = ["Case", "CaseInputs", "at_values", "checked_input", "expand", "inputs_for", "rng"]


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
                    formats=decl.formats or manifest.formats,
                    inputs=decl.inputs,
                    expected=decl.expected,
                    generator=decl.generator,
                    tier=decl.tier,
                    decl=decl,
                )
            )
    return tuple(cases)


def at_values(case: Case, name: str, values: Mapping[str, object]) -> Case:
    """``case`` run at other parameter values: the same generator, formats and
    tier, without its files (other values give other shapes). A benchmark
    point and a routed query run this way, over the default case.

    Args:
        case: The case to start from.
        name: The new case's name.
        values: Values over the case's own.

    Returns:
        The case, renamed, at ``{**case.values, **values}``.
    """
    merged = MappingProxyType({**case.values, **values})
    empty: Mapping[str, Path] = MappingProxyType({})
    decl = replace(case.decl, name=name, values=merged, inputs=empty, expected=empty)
    return replace(case, name=name, values=merged, inputs=empty, expected=empty, decl=decl)


def rng(kernel: str, case: str, tensor: str) -> np.random.Generator:
    """The random stream for one tensor of one case.

    Args:
        kernel: The kernel's name.
        case: The case's name, sweep point included.
        tensor: The input's name (``generator`` for a case generator's stream).

    Returns:
        A generator seeded from CRC-32 of ``"kernel/case/tensor"``: the same in
        every process (no ``hash()``), independent of every other tensor's.
    """
    return np.random.default_rng(
        np.random.SeedSequence(zlib.crc32(f"{kernel}/{case}/{tensor}".encode()))
    )


# --- the sources --------------------------------------------------------------------------


class _Bad(Exception):
    """Why a tensor from a file or a generator is refused; the caller says where."""


@dataclass(frozen=True)
class CaseInputs:
    """A case's input tensors.

    Attributes:
        tensors: Each input, by name, in declaration order, as the format holds
            it (int8 or int32 for an integer format, else float32).
        rounded: For an input from a file or a generator, how many of its values
            rounding to the format changed (E06-10's report), when any did.
    """

    tensors: Mapping[str, Array]
    rounded: Mapping[str, int]


def _checked(plan: CasePlan, name: str, value: object) -> Array:
    """``value`` as input ``name``: an int or float array of the planned shape, finite."""
    if not isinstance(value, np.ndarray):
        raise _Bad(f"{name} is {type(value).__name__}, not an array")
    if not (np.issubdtype(value.dtype, np.integer) or np.issubdtype(value.dtype, np.floating)):
        raise _Bad(f"{name} has dtype {value.dtype}, not an int or float one")
    shape = plan.shapes[name]
    if value.shape != shape:
        raise _Bad(f"{name} has shape {value.shape}, the case's is {shape}")
    if not np.all(np.isfinite(value)):
        raise _Bad(f"{name} holds a value that is not finite (NaN or infinity)")
    return value


def _rounded(values: Array, fmt: ElementFormat) -> tuple[Array, int]:
    """``values`` in the format, and how many of them that changed."""
    in_format = held(values, fmt)
    changed = int(np.count_nonzero(in_format.astype(np.float64) != values.astype(np.float64)))
    return in_format, changed


def checked_input(plan: CasePlan, name: str, value: object) -> Array:
    """``value`` as input ``name`` of ``plan``, checked as an ``--input`` file's
    tensor is and rounded to the format (a chain step's array, E06-15).

    Raises:
        ConfigurationError: It is no int or float array of the planned shape,
            or holds a value that is not finite.
    """
    try:
        return _rounded(_checked(plan, name, value), plan.format)[0]
    except _Bad as error:
        raise ConfigurationError(f"{plan.manifest.name}: input {error}") from None


def _loaded(plan: CasePlan, name: str, path: Path) -> tuple[Array, int]:
    """Input ``name`` from a ``.npy`` file, in the format."""
    try:
        value = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise _Bad(f"cannot read {path} as {name}: {error}") from None
    return _rounded(_checked(plan, name, value), plan.format)


def _from_generator(plan: CasePlan, case: Case) -> dict[str, tuple[Array, int]]:
    """What the case's ``generator`` makes, checked, in the format.

    The file is loaded afresh for each call (once per case and format), so an
    edit to it between runs is seen; it is a plain module with no state.
    """
    assert case.generator is not None
    key = f"cases.{case.decl.name}.generator"
    file, _, function = case.generator.partition(":")
    path = kernel_file(plan.manifest.directory, file)
    try:
        spec = importlib.util.spec_from_file_location(f"_ipu_case_generator_{path.stem}", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"no loader for {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as error:
        raise PlanError(
            plan.manifest, key, f"cannot load {file}: {type(error).__name__}: {error}"
        ) from error
    make = getattr(module, function, None)
    if not callable(make):
        raise PlanError(plan.manifest, key, f"{case.generator}: {file} has no function {function}")
    try:
        made = make(plan, rng(plan.manifest.name, case.name, "generator"))
    except Exception as error:
        raise PlanError(
            plan.manifest, key, f"{case.generator} raised {type(error).__name__}: {error}"
        ) from error
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
            out[name] = _rounded(_checked(plan, name, value), plan.format)
        except _Bad as error:
            raise PlanError(plan.manifest, key, f"{case.generator}: {error}") from None
    return out


def _generated(plan: CasePlan, case: Case, name: str) -> Array:
    """The default generator's values for input ``name``: uniform in the input's
    ``range`` for the format (closed: rounding can reach either end), else in
    ``[-1, 1]`` for a float format and the whole range of an integer one."""
    stream = rng(plan.manifest.name, case.name, name)
    shape = plan.shapes[name]
    bounds = plan.manifest.tensors[name].value_range.get(plan.format)
    if FORMATS[plan.format].is_float:
        lo, hi = bounds or (-1.0, 1.0)
        return held(stream.uniform(lo, hi, size=shape), plan.format)
    lo, hi = bounds or limits(plan.format)
    # The manifest checked that the range holds an integer of the format.
    values = stream.integers(math.ceil(lo), math.floor(hi), size=shape, endpoint=True)
    return held(values, plan.format)


def inputs_for(
    case: Case, plan: CasePlan, *, overrides: Mapping[str, Path] | None = None
) -> CaseInputs:
    """Every input tensor of ``case`` planned as ``plan``.

    Args:
        case: The case.
        plan: Its plan, for the format and the shapes.
        overrides: ``--input`` files, by tensor name; they win over the case.

    Returns:
        The inputs, and how many values of each file's or generator's tensor
        rounding to the format changed.

    Raises:
        ConfigurationError: An override names no input, or its file is
            unreadable, of the wrong shape or dtype, or holds a value that is
            not finite.
        PlanError: The same of a case's own file, or its generator fails to
            load or run, or returns something other than finite arrays of the
            declared inputs (located in ``kernel.toml``).
    """
    overrides = overrides or {}
    inputs = [t.name for t in plan.manifest.tensors.values() if t.role == "input"]
    unknown = sorted(set(overrides) - set(inputs))
    if unknown:
        raise ConfigurationError(
            f"--input {unknown[0]}=...: {unknown[0]!r} is no input of the kernel"
        )
    generated = _from_generator(plan, case) if case.generator else {}
    tensors: dict[str, Array] = {}
    rounded: dict[str, int] = {}
    for name in inputs:
        changed = 0
        if name in overrides:
            try:
                tensors[name], changed = _loaded(plan, name, overrides[name])
            except _Bad as error:
                raise ConfigurationError(f"--input {name}={overrides[name]}: {error}") from None
        elif name in case.inputs:
            key = f"cases.{case.decl.name}.inputs.{name}"
            try:
                tensors[name], changed = _loaded(plan, name, case.inputs[name])
            except _Bad as error:
                raise PlanError(plan.manifest, key, str(error)) from None
        elif name in generated:
            tensors[name], changed = generated[name]
        else:
            tensors[name] = _generated(plan, case, name)
        if changed:
            rounded[name] = changed
    return CaseInputs(MappingProxyType(tensors), MappingProxyType(rounded))
