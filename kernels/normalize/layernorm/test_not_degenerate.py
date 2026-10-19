"""LayerNorm outputs that a broken kernel could not produce (every entry runs
the checks on its own data).

The golden comparison alone passes a kernel whose output matches a reference
that is itself trivial on the data; these checks fail a constant output, a
pass-through of x, and output that is not centred and unit-variance over the
channels once gamma and beta are undone.
"""

from __future__ import annotations

import numpy as np
import pytest

from ipu_harness.cases import inputs_for
from ipu_harness.params import plan_case
from ipu_isa.formats import ElementFormat

FP32 = ElementFormat.FP32


def check_not_degenerate(actual, x, gamma, beta):
    """Raise AssertionError unless ``actual`` (``[channels, tokens]``) is a
    LayerNorm of ``x`` over the channels (axis 0)."""
    if not actual.std() > 0.1:
        raise AssertionError("output is (near) constant -- degenerate result")
    de_scaled = (actual - beta[:, None]) / gamma[:, None]
    tokens = actual.shape[1]
    np.testing.assert_allclose(
        de_scaled.mean(axis=0),
        np.zeros(tokens),
        atol=1e-4,
        err_msg="de-scaled output is not centred over the channel axis",
    )
    np.testing.assert_allclose(
        de_scaled.std(axis=0),
        np.ones(tokens),
        rtol=1e-3,
        err_msg="de-scaled output does not have unit variance over channels",
    )
    if np.allclose(actual, x, rtol=1e-2, atol=1e-2):
        raise AssertionError("output equals the input -- the kernel is a pass-through")


def _default(kernel, cases):
    (case,) = [c for c in cases if c.name == "default"]
    plan = plan_case(kernel, case.values, FP32, case=case.decl)
    return case, inputs_for(case, plan).tensors


def test_the_default_case_is_not_degenerate(kernel, cases, run_case):
    case, ins = _default(kernel, cases)
    result = run_case(case, FP32)
    assert result.passed, result.describe()
    y = np.asarray(result.outputs["y"])
    check_not_degenerate(y, ins["x"], ins["gamma"], ins["beta"])


def test_the_guard_fails_on_constant_input(kernel, cases):
    _, ins = _default(kernel, cases)
    x = np.full_like(ins["x"], 3.0)
    constant = np.broadcast_to(ins["beta"][:, None], x.shape)  # LayerNorm of x, no eps (Q57)
    with pytest.raises(AssertionError):  # de-scaled, the output is 0: not unit variance
        check_not_degenerate(constant, x, ins["gamma"], ins["beta"])
