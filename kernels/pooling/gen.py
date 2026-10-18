"""The pooling cases' data (E07-14), shared by the family's kernels.

Both draw ``x`` standard normal FP32 from ``default_rng(7)``, so the outputs
match the recorded ones bit for bit. Moving to the harness's seeded generator
is a separate change.
"""

import numpy as np

#: The cases' seed.
SEED = 7


def _normal(plan):
    return np.random.default_rng(SEED).standard_normal(plan.shapes["x"]).astype(np.float32)


def seeded_normal(plan, rng):
    """``x`` standard normal: the stride-2 kernels' cases."""
    return {"x": _normal(plan)}


def seeded_normal_minus_2(plan, rng):
    """``x`` standard normal minus 2, in FP32: the window kernels' cases."""
    return {"x": _normal(plan) - np.float32(2)}
