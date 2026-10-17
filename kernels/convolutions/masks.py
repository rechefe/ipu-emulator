"""The convolutions' mask regions (E07-16): ``bytes = "../masks.py:<fn>"``.

A mask is a bit per lane (1 keeps it), which no element value can carry, so
each is a scratch region's exact bytes: 8 slots of 16 bytes, then zeros to the
end of the region's row.
"""

from __future__ import annotations

from ipu_harness.params import CasePlan
from ipu_isa.formats import FORMATS
from ipu_isa.geometry import LANES


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


def border_mask(plan: CasePlan) -> bytes:
    """The universal 3x3 kernels' R_MASK blob (E07-17; ``build_border_mask_blob``
    in the ipu_apps family ``app.py``). ``128 // cols`` packed spatial rows share
    the 128 lanes. Slot 0 keeps every lane (interior taps), slot 3 zeros packed
    row 0 (the top border's kr = -1 taps), slot 6 zeros the last packed row (the
    bottom border's kr = +1 taps); left and right edges are mask shifts, not
    slots."""
    cols = int(plan.derived["cols"])
    last = LANES // cols - 1
    zeroed = {0: range(0), 3: range(cols), 6: range(last * cols, last * cols + cols)}
    blob = bytearray(FORMATS[plan.format].row_bytes)
    for slot, lanes in zeroed.items():
        for lane in range(LANES):
            if lane not in lanes:
                blob[slot * 16 + lane // 8] |= 1 << (lane % 8)
    return bytes(blob)
