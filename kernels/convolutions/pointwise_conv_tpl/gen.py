"""The data pattern of pointwise_conv_tpl's exact-zero check beyond the tolerance
(``test_pointwise_conv.py``); every other case takes the harness's seeded
streams."""

import numpy as np


def negative_bias(plan, rng):
    """Zero weights on an input of 1, each bias below -0.1: ReLU writes zeros."""
    p = plan.params
    i, o = p["in_channels"], p["out_channels"]
    return {
        "x": np.ones((i, p["height"], p["width"])),
        "w": np.zeros((o, i, 1, 1)),
        "b": -np.abs(rng.uniform(-1.0, 1.0, o)) - 0.1,
    }
