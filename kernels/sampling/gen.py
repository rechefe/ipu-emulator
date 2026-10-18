"""The sampling cases' data (E07-15), shared by both kernels.

``x`` is uniform in ``[-amplitude, amplitude)`` from ``default_rng(31)``, in
FP32, so the outputs match the recorded ones bit for bit.
"""

import numpy as np

#: The cases' seed.
SEED = 31


def legacy_uniform(plan, rng):
    """``x`` uniform in ``[-amplitude, amplitude)``, FP32."""
    amplitude = plan.params["amplitude"]
    state = np.random.default_rng(SEED)
    return {"x": state.uniform(-amplitude, amplitude, plan.shapes["x"]).astype(np.float32)}
