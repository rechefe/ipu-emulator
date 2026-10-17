"""The convolutions' mask regions (E07-16): ``bytes = "../masks.py:<fn>"``.

A mask is a bit per lane (1 keeps it), which no element value can carry, so
each is a scratch region's exact bytes: 8 slots of 16 bytes, then zeros to the
end of the region's row.
"""

from __future__ import annotations

from ipu_harness.params import CasePlan
from ipu_isa.formats import FORMATS


def first_layer_mask(plan: CasePlan) -> bytes:
    """``conv_first_layer``'s R_MASK blob: slot 0 keeps every lane, slot 1 zeros
    lane 0, slot 2 zeros lane 127. The program selects only slot 0 (its edges
    are zeroed by mask_shift), so slots 1 and 2 are written but never read."""
    blob = bytearray(FORMATS[plan.format].row_bytes)
    for slot in (0, 1, 2):
        blob[slot * 16 : slot * 16 + 16] = b"\xff" * 16
    blob[1 * 16 + 0] &= ~0x01  # slot 1, lane 0
    blob[2 * 16 + 15] &= ~0x80  # slot 2, lane 127
    return bytes(blob)
