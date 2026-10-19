"""The two data patterns of depthwise_conv_tpl's checks beyond the tolerance
(``test_depthwise_conv.py``); every other case takes the harness's seeded
streams. ``filters`` is each channel's bias, then its 9 taps."""

import numpy as np


def _filters(b, taps):
    return np.concatenate([b[:, None], taps], axis=1)


def negative_outputs(plan, rng):
    """Weights -1 on an input of 5, no bias: every sum is negative."""
    p = plan.params
    c = p["in_channels"]
    x = np.full((c, p["height"], p["width"]), 5.0)
    return {"x": x, "filters": _filters(np.zeros(c), -np.ones((c, 9)))}


def negative_bias(plan, rng):
    """Zero weights on an input of 1, each bias below -0.1: ReLU writes zeros."""
    p = plan.params
    c = p["in_channels"]
    b = -np.abs(rng.uniform(-1.0, 1.0, c)) - 0.1
    return {"x": np.ones((c, p["height"], p["width"])), "filters": _filters(b, np.zeros((c, 9)))}
