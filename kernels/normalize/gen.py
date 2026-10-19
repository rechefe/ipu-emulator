"""The normalize family's case data (E07-10): each kernel's old values, and the
packed LayerNorm's R_MASK rows (E07-23).

Each layernorm entry draws x, gamma and beta from ``numpy.random.RandomState``
with its own seed and recipe, over ``(channels, groups, tokens per group)``,
so the outputs match the recorded ones. ``l2_normalize_channels`` draws x
standard normal from ``default_rng(7)`` and zeroes column 0, so every case has
an all-zero column. The packed LayerNorms draw from ``RandomState(11)``.
"""

import numpy as np

from ipu_isa.formats import FORMATS
from ipu_kernels.pack import r_mask

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


def seeded_layernorm(plan, rng):
    recipe, seed = LAYERNORM.get(plan.manifest.name, LAYERNORM["layernorm_64x192"])
    channels, tokens = plan.params["channels"], plan.params["tokens"]
    groups = -(-tokens // LANES)
    draw = _randn if recipe == "randn" else _uniform
    old = np.random.RandomState(seed)
    x, gamma, beta = draw(old, channels, (channels, groups, tokens // groups))
    return {"x": x.reshape(channels, tokens), "gamma": gamma, "beta": beta}


def seeded_normal_zero_column(plan, rng):
    shape = (plan.params["channels"], plan.params["columns"])
    x = np.random.default_rng(7).standard_normal(shape).astype("<f4")
    x[:, 0] = 0
    return {"x": x}


def seeded_uniform_packed(plan, rng):
    """test_packed_layernorm_*.py's data: RandomState(11), x uniform(-3, 3),
    gamma uniform(0.5, 1.5), beta uniform(-0.5, 0.5); x as its packed rows."""
    channels, tokens = plan.params["channels"], plan.params["tokens"]
    old = np.random.RandomState(11)
    x = old.uniform(-3.0, 3.0, size=(channels, tokens)).astype(np.float32)
    gamma = old.uniform(0.5, 1.5, size=channels).astype(np.float32)
    beta = old.uniform(-0.5, 0.5, size=channels).astype(np.float32)
    return {"x": x.reshape(-1, LANES), "gamma": gamma, "beta": beta}


def window_mask(plan):
    """packed_layernorm's R_MASK row: each channel's window in its slot."""
    return r_mask.window_mask(plan.params["tokens"], FORMATS[plan.format].row_bytes)


def all_ones_mask(plan):
    """packed_layernorm's all-ones R_MASK row."""
    return r_mask.all_ones(FORMATS[plan.format].row_bytes)
