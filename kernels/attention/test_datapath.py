"""The score kernels against their datapath mirror (E07-12), bit for bit.

Each kernel's golden is the math reference (``op = "matmul"`` at FP32_ORDER).
This adds the second check of E06-09: the contraction over channels runs as
``MULT.RC.VE`` + ``ACC.ADD`` (``ACC.ADD.FIRST`` on channel 0), a float32
running sum per lane, so ``ipu_kernels.reference.datapath.acc_fold`` repeats
it exactly. A rounding change the tolerance tier would hide fails here; a bug
the mirror shares with the emulator still fails the math reference.

``y = a @ b`` with ``a`` ``[..., rows, d]`` and ``b`` ``[..., d, lanes]``: lane
``l`` of row ``r`` folds ``b[c, l] * a[r, c]`` over ``c``, i.e. ``acc_fold``
with ``P = b`` transposed (``[..., lanes, d]``) and ``V = a``.
"""

from __future__ import annotations

import numpy as np
import pytest

from ipu_harness.assembly import AssemblyCache
from ipu_harness.cases import expand, inputs_for
from ipu_harness.params import plan_case
from ipu_harness.registry import default_registry
from ipu_harness.runner import run_kernel
from ipu_isa.formats import ElementFormat
from ipu_kernels.reference.datapath import acc_fold

KERNELS = (
    "qk_scores_16x60",
    "qk_scores_64x48",
    "qk_scores_256x36",
    "attn_scores_km_16x60",
    "attn_scores_km_64x48",
    "attn_scores_km_256x36",
)

CACHE = AssemblyCache()


def _cases():
    registry = default_registry()
    for name in KERNELS:
        kernel = registry.kernel(name)
        for case in expand(kernel):
            if case.tier == "default":
                yield pytest.param(kernel, case, id=f"{name}-{case.name}")


def mirror(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``a @ b`` as the ACC datapath rounds it, batched over leading axes."""
    lead = a.shape[:-2]
    a3 = a.reshape(-1, *a.shape[-2:])
    b3 = b.reshape(-1, *b.shape[-2:])
    return acc_fold(np.swapaxes(b3, -1, -2), a3).reshape(*lead, a.shape[-2], b.shape[-1])


@pytest.mark.parametrize(("kernel", "case"), _cases())
def test_scores_equal_the_acc_datapath_mirror(kernel, case):
    fmt = ElementFormat.FP32
    inputs = inputs_for(case, plan_case(kernel, case.values, fmt, case=case.decl)).tensors
    result = run_kernel(kernel, case, fmt, cache=CACHE)
    assert result.passed, result.describe()
    np.testing.assert_array_equal(
        np.asarray(result.outputs["y"], dtype=np.float32), mirror(inputs["a"], inputs["b"])
    )
