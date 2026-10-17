"""Pass or fail, one way for every kernel (E06-10, Q80, Q82).

:func:`expected_for` gives each output's expected values: the case's
``expected`` file when it has one (Q82), else the reference's output (computed
from the inputs as the format holds them), rounded once to the format the
output is stored in (its element format, or its accumulator format) with
saturation. Quantization error therefore never counts against the kernel.
:func:`compare` holds the kernel's output to it at a named tier and, either
way, says where the outputs differ most and by how much.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ipu_harness._codec import Array, codes, held
from ipu_harness.cases import Case
from ipu_harness.manifest import TensorDecl
from ipu_harness.packing import stored_format
from ipu_harness.params import CasePlan, PlanError
from ipu_harness.tolerance import DEFAULT_TIER, Tier
from ipu_isa.formats import FORMATS, ElementFormat

__all__ = ["Comparison", "compare", "expected_for", "tier_for"]


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
        max_ulps: The largest distance in codes (8-bit float formats), else
            None; -1 when either value of an element is not a finite code.
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
    """The tier ``[golden] tolerance`` names for the format ``tensor`` is
    stored in (``plan.format``, or its accumulator format), else EXACT."""
    return plan.manifest.golden.tolerance.get(stored_format(tensor, plan.format), DEFAULT_TIER)


def _rounded(values: Array, tensor: TensorDecl, plan: CasePlan) -> Array:
    """``values`` as the format ``tensor`` is stored in holds them, as float64."""
    return held(np.asarray(values, dtype=np.float64), stored_format(tensor, plan.format)).astype(
        np.float64
    )


def expected_for(
    case: Case, plan: CasePlan, ref_out: Mapping[str, Array] | None
) -> dict[str, Array]:
    """Each output's expected values, as the stored format holds them.

    Args:
        case: The case (its ``expected`` files win over the reference).
        plan: Its plan.
        ref_out: The reference's float64 outputs (``reference.evaluate``), if
            the kernel has a reference.

    Returns:
        The expected values of every output that has a golden; an output with
        neither a file nor a reference output has none (a smoke run only).

    Raises:
        PlanError: An expected file is unreadable, not real numbers or not
            the output's shape; the reference did not make exactly the
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


def _ulps(got: Array, want: Array, fmt: ElementFormat) -> Array:
    """The distance in codes between two arrays of ``fmt`` values; infinite
    where either value is not a finite code of ``fmt``."""
    order = codes(fmt)

    def index(values: Array) -> Array:
        at = np.searchsorted(order, values)
        exact = order[np.minimum(at, len(order) - 1)] == values
        found: Array = np.where(exact, at, -1)
        return found

    g, w = index(got), index(want)
    distance: Array = np.where((g < 0) | (w < 0), np.inf, np.abs(g - w).astype(np.float64))
    return distance


def compare(output: str, got: Array, want: Array, tier: Tier, fmt: ElementFormat) -> Comparison:
    """Hold ``got`` to ``want`` at ``tier``.

    An element passes when ``|got - want| <= atol + rtol * |want|``, or, for a
    tier with ``ulps`` in an 8-bit float format, when both are codes of the
    format at most ``ulps`` codes apart. A NaN or infinity never passes.

    Args:
        output: The output's name.
        got: The kernel's values (unpacked).
        want: The expected values (:func:`expected_for`).
        tier: The tolerance.
        fmt: The format the values are stored in.

    Returns:
        The comparison, passed or not, with its worst element.

    Raises:
        ValueError: ``got`` and ``want`` differ in shape.
    """
    got64 = np.atleast_1d(np.asarray(got, dtype=np.float64))
    want64 = np.atleast_1d(np.asarray(want, dtype=np.float64))
    if got64.shape != want64.shape:
        raise ValueError(f"{output}: got shape {got64.shape}, want {want64.shape}")
    fp8 = FORMATS[fmt].is_float and FORMATS[fmt].element_bits == 8
    counts_codes = tier.ulps is not None and fp8
    if got64.size == 0:
        return Comparison(output, tier, True, 0.0, 0.0, 0 if counts_codes else None, (), 0.0, 0.0)
    with np.errstate(invalid="ignore"):
        diff = np.abs(got64 - want64)
        diff[~np.isfinite(diff)] = np.inf  # a NaN is as far off as an infinity
        rel = np.divide(diff, np.abs(want64), out=np.zeros_like(diff), where=want64 != 0)
        ulps: Array | None = None
        if counts_codes:
            ulps = _ulps(got64, want64, fmt)
            excess = ulps - tier.ulps
        else:
            excess = diff - (tier.atol + tier.rtol * np.abs(want64))
    passed = bool(np.all(excess <= 0))
    worst = np.unravel_index(int(np.argmax(excess if not passed else diff)), diff.shape)
    index = tuple(int(i) for i in worst)
    max_ulps = None
    if ulps is not None:
        max_ulps = -1 if np.isinf(ulps).any() else int(ulps.max())
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
