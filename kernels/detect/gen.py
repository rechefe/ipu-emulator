"""The detect cases' data (E07-15), shared by the family's kernels.

``x`` is standard normal FP32 from ``default_rng(7)``, so the outputs match
the recorded ones bit for bit. Moving to the harness's seeded generator is a
separate change.
"""

import numpy as np

#: The cases' seed.
SEED = 7


def seeded_normal(plan, rng):
    """``x`` standard normal FP32."""
    return {"x": np.random.default_rng(SEED).standard_normal(plan.shapes["x"]).astype(np.float32)}
