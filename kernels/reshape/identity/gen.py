"""The ipu_apps identity cases' values: ``arange(rows * 128) - 128``, row-major."""

import numpy as np


def arange(plan, rng):
    rows = plan.params["rows"]
    values = np.arange(rows * 128, dtype=np.float64) - 128
    return {"x": values.reshape(rows, 128)}
