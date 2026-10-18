"""The old detect cases' data (E07-15), shared by the family's kernels.

``x`` is drawn as ``case_support.random_values`` did: standard normal FP32
from ``default_rng(7)``, so the migration's outputs compare bit for bit with
the old harness's. Moving to the harness's seeded generator is a separate
change.
"""

import numpy as np

#: The old cases' seed.
SEED = 7


def legacy_normal(plan, rng):
    """``x`` standard normal FP32."""
    return {"x": np.random.default_rng(SEED).standard_normal(plan.shapes["x"]).astype(np.float32)}
