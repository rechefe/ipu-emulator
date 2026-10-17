"""score_threshold's ``T * tau`` row (E07-15): ``T * tau`` in every lane, as
the FP32 product of the FP32 ``temperature`` and ``threshold`` (the old case's
arithmetic; a ``fill`` expression would round the exact product once)."""

import numpy as np


def ttau(plan):
    t = np.float32(plan.params["temperature"])
    tau = np.float32(plan.params["threshold"])
    return np.full(128, t * tau, dtype="<f4").tobytes()
