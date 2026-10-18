"""The reshape family's case data (E07-05), shared by the fold and unfold
kernels, and the packed seams' R_MASK row (E07-23).

Each kernel's default case keeps the old data: a ``[c, h, w]``
tensor from ``RandomState(seed)``, uniform in [-1, 1), in FP32
(``reshape/cases.py:spatial_tensor``), with the old seed per kernel.
"""

import numpy as np

from ipu_isa.formats import FORMATS
from ipu_kernels.pack import r_mask

#: Each kernel's old seed.
SEEDS = {
    "unfold_8x8x240": 0x058,
    "unfold_16x16x192": 0x016,
    "unfold_32x32x144": 0x032,
    "fold_8x8x240": 0x05A,
    "fold_16x16x192": 0x018,
    "fold_32x32x144": 0x034,
}


def _spatial(plan):
    c, h, w = plan.params["c"], plan.params["h"], plan.params["w"]
    old = np.random.RandomState(SEEDS[plan.manifest.name])
    return old.uniform(-1.0, 1.0, size=(c, h, w)).astype(np.float32).astype(np.float64)


def spatial(plan, rng):
    """An unfold's input ``x``."""
    return {"x": _spatial(plan)}


def phases(plan, rng):
    """A fold's input ``x``: the four stride-2 phases of the tensor, ``[4, c, tokens]``."""
    x = _spatial(plan)
    c = x.shape[0]
    return {"x": np.stack([x[:, i::2, j::2].reshape(c, -1) for i in range(2) for j in range(2)])}


def window_mask(plan):
    """packed_pack's and packed_unpack's R_MASK row: each channel's window in its slot."""
    return r_mask.window_mask(plan.params["tokens"], FORMATS[plan.format].row_bytes)
