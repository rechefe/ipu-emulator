"""Pass or fail, one way for every kernel (E06-10, Q80, Q82).

:func:`expected_for` gives each output's expected values: the case's
``expected`` file when it has one (Q82; or the values its ``"file.py:fn"``
returns for the plan), else the reference's output (computed from the inputs
as the variant holds them), held once by the variant the output is stored in
(FP32: rounded to float32; NX8: packed and quantized row by row, ADR 0007).
Quantization error therefore never counts against the kernel.
:func:`compare` holds the kernel's output to it at a named tier and, either
way, says where the outputs differ most and by how much. In NX8 the
``NX8_STEP`` tier measures the distance in steps of the golden row's grid
(:func:`nx8_steps`): a row's scale follows its largest value, which the
kernel's accumulation order can move by one step.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ipu_harness._codec import Array, row_grid
from ipu_harness._plugins import kernel_file, load_file
from ipu_harness.cases import Case
from ipu_harness.manifest import TensorDecl
from ipu_harness.packing import held, logical, pack, transform_for
from ipu_harness.params import CasePlan, PlanError
from ipu_harness.tolerance import DEFAULT_TIER, Tier
from ipu_harness.variants import Variant
from ipu_isa.geometry import LANES

__all__ = ["Comparison", "compare", "expected_for", "nx8_steps", "tier_for"]


@dataclass(frozen=True)
class Comparison:
    """One output held to its expected values.

    Attributes:
        output: The output's name.
        tier: The tolerance it was held to.
        passed: Whether every element is within the tier.
        max_abs: The largest absolute difference.
        max_rel: The largest difference relative to the expected value, over
            the elements whose expected value is not zero.
        max_ulps: The largest distance in NX8 steps (a tier that counts
            them), rounded up, else None; -1 when a value is not finite.
        worst_index: Where the element that is furthest out of tolerance is
            (the largest difference when all pass).
        got: The kernel's value there.
        want: The expected value there.
    """

    output: str
    tier: Tier
    passed: bool
    max_abs: float
    max_rel: float
    max_ulps: int | None
    worst_index: tuple[int, ...]
    got: float
    want: float

    def describe(self) -> str:
        """One line for a report."""
        verdict = "PASS" if self.passed else "FAIL"
        ulps = "" if self.max_ulps is None else f" max ulps={self.max_ulps}"
        return (
            f"{self.output}: {verdict} at {self.tier.name} (max|err|={self.max_abs:g} "
            f"max rel={self.max_rel:g}{ulps}); worst at {list(self.worst_index)}: "
            f"got {self.got!r}, want {self.want!r}"
        )


def tier_for(plan: CasePlan, tensor: TensorDecl) -> Tier:
    """The tier ``[golden] tolerance`` names for the case's variant (its own
    key, or its mode's), else EXACT."""
    return plan.manifest.golden.tolerance.get(plan.format, DEFAULT_TIER)


def _rounded(values: Array, tensor: TensorDecl, plan: CasePlan) -> Array:
    """``values`` as ``tensor``'s region holds them in the case's variant."""
    return held(tensor, np.asarray(values, dtype=np.float64), plan)


def nx8_steps(tensor: TensorDecl, want: Array, plan: CasePlan) -> Array:
    """One NX8 step at each element of ``want`` (an output's golden, as held):
    the larger gap to a neighbour on the grid of the golden row it sits in.

    Raises:
        ValueError: ``plan`` runs in FP32 (no grid).
    """
    values = np.asarray(want, dtype=np.float64)
    packed = pack(tensor, values, plan)
    if packed.metadata is None:
        raise ValueError(f"{tensor.name}: an FP32 output has no NX8 steps")
    rows = np.asarray(transform_for(tensor, plan).to_rows(values, plan)).reshape(-1, LANES)
    steps = np.empty(rows.shape, dtype=np.float64)
    for index, metadata in enumerate(packed.metadata):
        grid = row_grid(plan.format, metadata)
        at = np.clip(np.searchsorted(grid, rows[index]), 0, len(grid) - 1)
        above = grid[np.minimum(at + 1, len(grid) - 1)] - grid[at]
        below = grid[at] - grid[np.maximum(at - 1, 0)]
        steps[index] = np.maximum(above, below)
    return logical(tensor, steps, plan)


def _from_function(source: str, plan: CasePlan, key: str) -> Array:
    """The values ``"file.py:fn"``'s ``fn(plan)`` returns (the kernel folder's
    file, else its family folder's), as an array."""
    file, _, function = source.partition(":")
    path = kernel_file(plan.manifest.directory, file)
    try:
        module = load_file(path)
    except Exception as error:
        raise PlanError(
            plan.manifest, key, f"cannot load {file}: {type(error).__name__}: {error}"
        ) from error
    fn = getattr(module, function, None)
    if not callable(fn):
        raise PlanError(plan.manifest, key, f"{source}: {file} has no function {function}")
    try:
        return np.asarray(fn(plan))
    except Exception as error:
        raise PlanError(
            plan.manifest, key, f"{source} raised {type(error).__name__}: {error}"
        ) from error


def expected_for(
    case: Case, plan: CasePlan, ref_out: Mapping[str, Array] | None
) -> dict[str, Array]:
    """Each output's expected values, as the stored format holds them.

    Args:
        case: The case (its ``expected`` files or functions win over the
            reference).
        plan: Its plan.
        ref_out: The reference's float64 outputs (``reference.evaluate``), if
            the kernel has a reference.

    Returns:
        The expected values of every output that has a golden; an output with
        neither a file nor a reference output has none (a smoke run only).

    Raises:
        PlanError: An expected file is unreadable (or its function fails to
            load or raises), not real numbers or not the output's shape; the reference did not make exactly the
            declared outputs, or made one the wrong shape; or an expected
            value is not finite (located at ``cases.<name>.expected.<output>``
            or ``golden``).
    """
    outputs = [t for t in plan.manifest.tensors.values() if t.role == "output"]
    if ref_out is not None and set(ref_out) != {t.name for t in outputs}:
        raise PlanError(
            plan.manifest,
            "golden",
            f"the reference made {sorted(ref_out)}, not the outputs "
            f"{sorted(t.name for t in outputs)}",
        )
    out: dict[str, Array] = {}
    for tensor in outputs:
        shape = plan.shapes[tensor.name]
        if tensor.name in case.expected:
            key = f"cases.{case.decl.name}.expected.{tensor.name}"
            path = case.expected[tensor.name]
            if isinstance(path, str):
                values = _from_function(path, plan, key)
            else:
                try:
                    values = np.load(path, allow_pickle=False)
                except (OSError, ValueError) as error:
                    raise PlanError(plan.manifest, key, f"cannot read {path}: {error}") from None
            real = np.issubdtype(values.dtype, np.floating) or np.issubdtype(
                values.dtype, np.integer
            )
            if values.shape != shape or not real:
                raise PlanError(
                    plan.manifest, key, f"{path} holds {values.dtype} {values.shape}, not {shape}"
                )
        elif ref_out is not None:
            key = "golden"
            values = np.asarray(ref_out[tensor.name])
            if values.shape != shape:
                raise PlanError(
                    plan.manifest,
                    key,
                    f"the reference made {tensor.name} {values.shape}, not {shape}",
                )
        else:
            continue
        if not np.all(np.isfinite(values)):
            raise PlanError(plan.manifest, key, f"expected {tensor.name} holds a non-finite value")
        out[tensor.name] = _rounded(values, tensor, plan)
    return out


def compare(
    output: str,
    got: Array,
    want: Array,
    tier: Tier,
    fmt: Variant,
    steps: Array | None = None,
) -> Comparison:
    """Hold ``got`` to ``want`` at ``tier``.

    An element passes when ``|got - want| <= atol + rtol * |want|``, or, for a
    tier with ``ulps`` in NX8, when it is at most ``ulps`` NX8 steps
    (``steps``, from :func:`nx8_steps`) away. A NaN or infinity never passes.

    Args:
        output: The output's name.
        got: The kernel's values (unpacked).
        want: The expected values (:func:`expected_for`).
        tier: The tolerance.
        fmt: The case's variant.
        steps: One NX8 step at each element (NX8 and a tier that counts them).

    Returns:
        The comparison, passed or not, with its worst element.

    Raises:
        ValueError: ``got`` and ``want`` differ in shape, or a tier counts
            steps in NX8 and ``steps`` is missing.
    """
    got64 = np.atleast_1d(np.asarray(got, dtype=np.float64))
    want64 = np.atleast_1d(np.asarray(want, dtype=np.float64))
    if got64.shape != want64.shape:
        raise ValueError(f"{output}: got shape {got64.shape}, want {want64.shape}")
    counts_steps = tier.counts_steps and fmt.nx8
    if counts_steps and steps is None:
        raise ValueError(f"{output}: {tier.name} counts NX8 steps; none given")
    if got64.size == 0:
        return Comparison(output, tier, True, 0.0, 0.0, 0 if counts_steps else None, (), 0.0, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        diff = np.abs(got64 - want64)
        diff[~np.isfinite(diff)] = np.inf  # a NaN is as far off as an infinity
        rel = np.divide(diff, np.abs(want64), out=np.zeros_like(diff), where=want64 != 0)
        distance: Array | None = None
        if counts_steps:
            step = np.atleast_1d(np.asarray(steps, dtype=np.float64)).reshape(diff.shape)
            distance = np.where(diff == 0, 0.0, diff / step)
            excess = distance - (tier.ulps or 0)
        else:
            excess = diff - (tier.atol + tier.rtol * np.abs(want64))
    passed = bool(np.all(excess <= 0))
    worst = np.unravel_index(int(np.argmax(excess if not passed else diff)), diff.shape)
    index = tuple(int(i) for i in worst)
    max_ulps = None
    if distance is not None:
        max_ulps = -1 if not np.isfinite(distance).all() else int(np.ceil(distance.max()))
    return Comparison(
        output=output,
        tier=tier,
        passed=passed,
        max_abs=float(diff.max()),
        max_rel=float(rel.max()),
        max_ulps=max_ulps,
        worst_index=index,
        got=float(got64[index]),
        want=float(want64[index]),
    )
