"""vector_tour's data: the cases' inputs, and the R_MASK slots.

``seeded_steps`` draws from ``default_rng(16)``, the old cases' seed.
"""

import numpy as np

#: The old cases' seed.
SEED = 16

#: The data's grid: every NX8 variant holds multiples of 0.25 in [-2, 2] exactly.
STEP = 0.25

#: R_MASK slots, 16 bytes each, bit i = lane i: all lanes, even lanes, lower half.
MASK_SLOTS = (b"\xff" * 16, b"\x55" * 16, b"\xff" * 8 + b"\x00" * 8)

MASK_IMAGE_BYTES = 128


def seeded_steps(plan, rng):
    """a, w, c: integers in [-8, 8] times 0.25."""
    state = np.random.default_rng(SEED)
    return {
        name: state.integers(-8, 9, size=plan.shapes[name]).astype(np.float64) * STEP
        for name in ("a", "w", "c")
    }


def mask_slots(plan):
    """The mask row's bytes: the three slots, then zeros."""
    return b"".join(MASK_SLOTS).ljust(MASK_IMAGE_BYTES, b"\x00")
