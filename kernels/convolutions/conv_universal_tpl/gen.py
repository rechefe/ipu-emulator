"""The two data patterns of conv_universal_tpl's checks beyond the tolerance
(``test_conv_universal.py``); every other case takes the harness's seeded
streams. ``filters`` is the weights, then the bias column."""

import numpy as np


def _filters(w, b):
    return np.concatenate([w.reshape(w.shape[0], -1), b[:, None]], axis=1)


def negative_sums(plan, rng):
    """Weights -1 on an input of 5, no bias: every sum is negative."""
    p = plan.params
    i, o = p["in_channels"], p["out_channels"]
    x = np.full((i, p["height"], p["width"]), 5.0)
    return {"x": x, "filters": _filters(-np.ones((o, i, 3, 3)), np.zeros(o))}


def negative_bias(plan, rng):
    """Zero weights on an input of 1, each bias below -0.1: ReLU writes zeros."""
    p = plan.params
    i, o = p["in_channels"], p["out_channels"]
    b = -np.abs(rng.uniform(-1.0, 1.0, o)) - 0.1
    return {
        "x": np.ones((i, p["height"], p["width"])),
        "filters": _filters(np.zeros((o, i, 3, 3)), b),
    }
