"""The ipu_apps residual_add cases' data: ``RandomState(42)``, ``a`` uniform in
[-4, 4), then ``b`` in [-6, 6), FP32, one channel per 128-element row."""

import numpy as np


def legacy_uniform(plan, rng):
    n = plan.params["num_channels"]
    old = np.random.RandomState(42)
    a = old.uniform(-4.0, 4.0, size=(n, 128)).astype(np.float32)
    b = old.uniform(-6.0, 6.0, size=(n, 128)).astype(np.float32)
    return {"a": a.astype(np.float64), "b": b.astype(np.float64)}
