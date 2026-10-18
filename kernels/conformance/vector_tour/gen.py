"""vector_tour's data: the cases' inputs, and the R_MASK slots.

``seeded_steps`` draws from ``default_rng(16)``, so the outputs match the
recorded ones bit for bit.
"""

import numpy as np

#: The old cases' seed.
SEED = 16

#: R_MASK slots, 16 bytes each, bit i = lane i: all lanes, even lanes, lower half.
MASK_SLOTS = (b"\xff" * 16, b"\x55" * 16, b"\xff" * 8 + b"\x00" * 8)

MASK_IMAGE_BYTES = 128


def seeded_steps(plan, rng):
    """a, w, c: integers in [-8, 8] in INT8, those times 0.25 in a float format."""
    state = np.random.default_rng(SEED)
    step = 1.0 if plan.format.value == "int8" else 0.25
    return {
        name: state.integers(-8, 9, size=plan.shapes[name]).astype(np.float64) * step
        for name in ("a", "w", "c")
    }


def mask_slots(plan):
    """The mask row's bytes: the three slots, then zeros."""
    return b"".join(MASK_SLOTS).ljust(MASK_IMAGE_BYTES, b"\x00")
