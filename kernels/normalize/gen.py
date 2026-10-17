"""The normalize family's case data (E07-10): each kernel's old ipu_apps values.

Each layernorm entry drew x, gamma and beta from ``numpy.random.RandomState``
with its own seed and recipe (the old ``normalize/cases.py``), over
``(channels, groups, tokens per group)``; keeping them keeps the outputs equal
to the old ones. ``l2_normalize_channels`` drew x from ``default_rng(7)``
(``case_support.random_values``) and zeroed column 0, so every case has an
all-zero column.
"""

import numpy as np

LANES = 128

#: Each layernorm kernel's old recipe and seed (the template's own default
#: case takes layernorm_64x192's).
LAYERNORM = {
    "layernorm_128x16": ("randn", 42),
    "layernorm_16x240": ("uniform", 0x16240),
    "layernorm_64x192": ("uniform", 0x64192),
    "layernorm_256x144": ("randn", 42),
}


def _randn(rng, channels, shape):
    """Normal x, gamma near 1, small beta."""
    x = rng.randn(*shape).astype(np.float32)
    gamma = rng.randn(channels).astype(np.float32) * 0.5 + 1.0
    beta = rng.randn(channels).astype(np.float32) * 0.1
    return x, gamma, beta


def _uniform(rng, channels, shape):
    """x in [-4, 4), gamma in [0.5, 1.5), beta in [-0.5, 0.5)."""
    x = rng.uniform(-4.0, 4.0, size=shape).astype(np.float32)
    gamma = rng.uniform(0.5, 1.5, size=channels).astype(np.float32)
    beta = rng.uniform(-0.5, 0.5, size=channels).astype(np.float32)
    return x, gamma, beta


def legacy_layernorm(plan, rng):
    recipe, seed = LAYERNORM.get(plan.manifest.name, LAYERNORM["layernorm_64x192"])
    channels, tokens = plan.params["channels"], plan.params["tokens"]
    groups = -(-tokens // LANES)
    draw = _randn if recipe == "randn" else _uniform
    old = np.random.RandomState(seed)
    x, gamma, beta = draw(old, channels, (channels, groups, tokens // groups))
    return {"x": x.reshape(channels, tokens), "gamma": gamma, "beta": beta}


def legacy_l2(plan, rng):
    shape = (plan.params["channels"], plan.params["columns"])
    x = np.random.default_rng(7).standard_normal(shape).astype("<f4")
    x[:, 0] = 0
    return {"x": x}
