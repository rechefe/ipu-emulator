"""The conv_universal cases' ipu_apps data (E07-17), for the proof only.

``generator = "../gen.py:<fn>"``: convolutions/cases.py's ``_data`` and its
data modes, seed 0 and draw order (weights, input, then the bias when the case
has one), so the outputs compare bit for bit with the old harness's. The next
commit moves the cases to the harness's seeded generator and deletes this file.
"""

import numpy as np


def _universal(plan, mode="random"):
    p = plan.params
    i, o = p["in_channels"], p["out_channels"]
    r = np.random.RandomState(0)
    w = (r.randn(o, i, 3, 3) * 0.2).astype(np.float32)
    x = (r.randn(i, p["height"], p["width"]) * 0.5).astype(np.float32)
    b = (r.randn(o) * 0.3).astype(np.float32) if p["has_bias"] else np.zeros(o, np.float32)
    if mode == "negative":
        w, x = -np.ones_like(w), np.full_like(x, 5.0)
    elif mode == "negative_bias":
        w, x, b = np.zeros_like(w), np.ones_like(x), -np.abs(b) - 0.1
    return {"x": x, "filters": np.concatenate([w.reshape(o, i * 9), b[:, None]], axis=1)}


def legacy_universal(plan, rng):
    return _universal(plan)


def legacy_universal_negative(plan, rng):
    return _universal(plan, "negative")


def legacy_universal_negative_bias(plan, rng):
    return _universal(plan, "negative_bias")
