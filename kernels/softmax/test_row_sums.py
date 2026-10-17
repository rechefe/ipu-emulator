"""Every softmax sums to 1 (E07-11): each kernel's default-tier cases, run, and
each output's sums along the reduced axis held to 1 +- 1e-5.

The golden comparison holds each element to the reference at TRANSCENDENT; this
is the ``ipu_apps`` cases' second check, on the whole distribution: an error
that every element hides within its tolerance still shows in a sum of up to
2000 of them.
"""

from __future__ import annotations

import numpy as np
import pytest

from ipu_harness.assembly import AssemblyCache
from ipu_harness.cases import expand
from ipu_harness.registry import default_registry
from ipu_harness.runner import run_kernel
from ipu_isa.formats import ElementFormat

KERNELS = (
    "softmax_columns",
    "softmax_columns_packed",
    "softmax_rows",
    "softmax_rows_long",
    "softmax_rows_partial",
)

CACHE = AssemblyCache()


def _cases():
    registry = default_registry()
    for name in KERNELS:
        kernel = registry.kernel(name)
        for case in expand(kernel):
            if case.tier == "default":
                yield pytest.param(kernel, case, id=f"{name}-{case.name}")


@pytest.mark.parametrize(("kernel", "case"), _cases())
def test_each_softmax_sums_to_one(kernel, case):
    result = run_kernel(kernel, case, ElementFormat.FP32, cache=CACHE)
    assert result.passed, result.describe()
    sums = np.asarray(result.outputs["y"]).sum(axis=int(case.values["dim"]))
    deviation = float(np.abs(sums - 1).max())
    assert deviation <= 1e-5, f"{kernel.name} {case.name}: sums differ from 1 by {deviation:.3g}"
