"""The old depth_to_space cases' values: ``arange`` over ``[r * r, h, w]``."""

import numpy as np


def arange(plan, rng):
    r, h, w = plan.params["upscale_factor"], plan.params["h"], plan.params["w"]
    return {"x": np.arange(r * r * h * w, dtype=np.float64).reshape(r * r, h, w)}
