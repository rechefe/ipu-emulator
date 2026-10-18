"""The old pooling cases' data (E07-14), shared by the family's kernels.

Both draw ``x`` as ``case_support.random_values`` did: standard normal FP32
from ``default_rng(7)``, so the migration's outputs compare bit for bit with
the old harness's. Moving to the harness's seeded generator is a separate
change.
"""

import numpy as np

#: The old cases' seed.
SEED = 7


def _normal(plan):
    return np.random.default_rng(SEED).standard_normal(plan.shapes["x"]).astype(np.float32)


def legacy_normal(plan, rng):
    """``x`` standard normal: the stride-2 kernels' cases."""
    return {"x": _normal(plan)}


def legacy_window(plan, rng):
    """``x`` standard normal minus 2, in FP32: the window kernels' cases."""
    return {"x": _normal(plan) - np.float32(2)}
