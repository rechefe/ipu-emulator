"""The attention kernels against their datapath mirrors (E07-12, E07-13), bit for bit.

Each kernel's golden is the math reference (``op = "matmul"``). This adds the
second check of E06-09, ``ipu_kernels.reference.datapath``: a rounding change
the tolerance tier would hide fails here; a bug the mirror shares with the
emulator still fails the math reference.

* The score kernels and ``attn_v_bcast_*`` contract with ``MULT.RC.VE`` +
  ``ACC.ADD`` (``ACC.ADD.FIRST`` first), a float32 running sum per lane:
  ``acc_fold``. ``y = a @ b`` with ``a`` ``[..., rows, k]`` and ``b``
  ``[..., k, lanes]``: lane ``l`` of row ``r`` folds ``b[c, l] * a[r, c]``
  over ``c``, i.e. ``acc_fold`` with ``P = b`` transposed and ``V = a``.
* ``attn_v_*`` contracts with ``MULT.RC.VV`` + ``AGG.SUM[.FIRST]`` over each
  128-key chunk: ``agg_fold``. ``y = a @ b^T`` with ``b`` = P query-major
  ``[h, i, s]`` and ``a`` = V ``[h, t, s]``: ``agg_fold(P = b, V = a)``.
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
from ipu_kernels.reference.datapath import acc_mirror, agg_mirror

#: Each kernel's datapath mirror.
KERNELS = {
    "qk_scores_16x60": acc_mirror,
    "qk_scores_64x48": acc_mirror,
    "qk_scores_256x36": acc_mirror,
    "attn_scores_km_16x60": acc_mirror,
    "attn_scores_km_64x48": acc_mirror,
    "attn_scores_km_256x36": acc_mirror,
    "attn_v_bcast_36": acc_mirror,
    "attn_v_bcast_48": acc_mirror,
    "attn_v_bcast_60": acc_mirror,
    "attn_v_16x60": agg_mirror,
    "attn_v_64x48": agg_mirror,
    "attn_v_256x36": agg_mirror,
}

CACHE = AssemblyCache()


def _cases():
    registry = default_registry()
    kernels = [registry.kernel(name) for name in KERNELS]
    return [
        pytest.param(kernel, case, id=f"{kernel.name}-{case.name}")
        for kernel in kernels
        for case in expand(kernel)
        if case.tier == "default"
    ]


@pytest.mark.parametrize(("kernel", "case"), _cases())
def test_each_kernel_equals_its_datapath_mirror(kernel, case):
    fmt = ElementFormat.FP32
    inputs = inputs_for(case, plan_case(kernel, case.values, fmt, case=case.decl)).tensors
    result = run_kernel(kernel, case, fmt, cache=CACHE)
    assert result.passed, result.describe()
    np.testing.assert_array_equal(
        np.asarray(result.outputs["y"], dtype=np.float32),
        KERNELS[kernel.name](inputs["a"], inputs["b"]),
    )
