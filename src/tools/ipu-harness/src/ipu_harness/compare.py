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
        max_rel: The largest difference relative to the expected value.
        max_ulps: The largest distance in codes (8-bit float formats), else None.
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


def tier_for(plan: CasePlan) -> Tier:
    """The tier ``[golden] tolerance`` names for ``plan.format``, else EXACT."""
    return plan.manifest.golden.tolerance.get(plan.format, DEFAULT_TIER)


def _rounded(values: Array, tensor: TensorDecl, plan: CasePlan) -> Array:
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
        PlanError: An expected file is unreadable or not the output's shape,
            or the reference gave an output the wrong shape (located at
            ``cases.<name>.expected.<output>`` or ``golden``).
    """
    out: dict[str, Array] = {}
    for tensor in plan.manifest.tensors.values():
        if tensor.role != "output":
            continue
        shape = plan.shapes[tensor.name]
        if tensor.name in case.expected:
            key = f"cases.{case.decl.name}.expected.{tensor.name}"
            path = case.expected[tensor.name]
            try:
                values = np.load(path, allow_pickle=False)
            except (OSError, ValueError) as error:
                raise PlanError(plan.manifest, key, f"cannot read {path}: {error}") from None
            if values.shape != shape or not np.issubdtype(values.dtype, np.number):
                raise PlanError(
                    plan.manifest, key, f"{path} holds {values.dtype} {values.shape}, not {shape}"
                )
        elif ref_out is not None and tensor.name in ref_out:
            values = ref_out[tensor.name]
            if values.shape != shape:
                raise PlanError(
                    plan.manifest,
                    "golden",
                    f"the reference made {tensor.name} {values.shape}, not {shape}",
                )
        else:
            continue
        out[tensor.name] = _rounded(values, tensor, plan)
    return out


def _ulps(got: Array, want: Array, fmt: ElementFormat) -> Array:
    """The distance in codes between two arrays of ``fmt`` values."""
    order = codes(fmt)
    distance: Array = np.abs(np.searchsorted(order, got) - np.searchsorted(order, want))
    return distance


def compare(output: str, got: Array, want: Array, tier: Tier, fmt: ElementFormat) -> Comparison:
    """Hold ``got`` to ``want`` at ``tier``.

    An element passes when ``|got - want| <= atol + rtol * |want|``, or, for a
    tier with ``ulps`` in an 8-bit float format, when the two are at most
    ``ulps`` codes apart.

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
    got64, want64 = np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64)
    if got64.shape != want64.shape:
        raise ValueError(f"{output}: got shape {got64.shape}, want {want64.shape}")
    diff = np.abs(got64 - want64)
    rel = np.divide(diff, np.abs(want64), out=np.zeros_like(diff), where=want64 != 0)
    ulps: Array | None = None
    if tier.ulps is not None and FORMATS[fmt].is_float and FORMATS[fmt].element_bits == 8:
        ulps = _ulps(got64, want64, fmt)
        excess = ulps.astype(np.float64) - tier.ulps
    else:
        excess = diff - (tier.atol + tier.rtol * np.abs(want64))
    passed = bool(np.all(excess <= 0))
    worst = np.unravel_index(int(np.argmax(excess if not passed else diff)), diff.shape)
    index = tuple(int(i) for i in worst)
    return Comparison(
        output=output,
        tier=tier,
        passed=passed,
        max_abs=float(diff.max(initial=0.0)),
        max_rel=float(rel.max(initial=0.0)),
        max_ulps=None if ulps is None else int(ulps.max(initial=0)),
        worst_index=index,
        got=float(got64[index]),
        want=float(want64[index]),
    )
