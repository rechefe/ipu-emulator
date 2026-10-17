"""vector_tour's data: the ``ipu_apps`` cases' inputs, and the R_MASK slots.

``legacy_steps`` replays the old cases' ``default_rng(16)`` draws, so the
migration's outputs compare bit for bit with the old harness's.
"""

import numpy as np

#: The ``ipu_apps`` cases' seed.
SEED = 16

#: R_MASK slots, 16 bytes each, bit i = lane i: all lanes, even lanes, lower half.
MASK_SLOTS = (b"\xff" * 16, b"\x55" * 16, b"\xff" * 8 + b"\x00" * 8)

ROW_BYTES = 128


def legacy_steps(plan, rng):
    """a, w, c: integers in [-8, 8] in INT8, those times 0.25 in a float format."""
    state = np.random.default_rng(SEED)
    step = 1.0 if plan.format.value == "int8" else 0.25
    return {
        name: state.integers(-8, 9, size=plan.shapes[name]).astype(np.float64) * step
        for name in ("a", "w", "c")
    }


def mask_slots(plan):
    """The mask row's bytes: the three slots, then zeros."""
    return b"".join(MASK_SLOTS).ljust(ROW_BYTES, b"\x00")
