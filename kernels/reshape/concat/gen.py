"""The ipu_apps concat cases' values: ``RandomState(0xC047)``, a then b, uniform [-1, 1) in FP32."""

import numpy as np


def legacy_uniform(plan, rng):
    old = np.random.RandomState(0xC047)
    a = old.uniform(-1.0, 1.0, size=(plan.params["c_a"], 128)).astype(np.float32)
    b = old.uniform(-1.0, 1.0, size=(plan.params["c_b"], 128)).astype(np.float32)
    return {"a": a.astype(np.float64), "b": b.astype(np.float64)}
