"""The softmax kernels' resident rows that no single fill value makes (E07-11).

Each function gives a two-row scratch region's starting bytes (``bytes =
"../scratch.py:<fn>"`` in ``kernel.toml``), in FP32, the kernels' one
format: the ``.asm`` reads the second row as the first row's neighbour, through
no register of its own, so both rows are one region.
"""

from __future__ import annotations

import numpy as np

from ipu_isa.geometry import LANES
from ipu_kernels.softmax.pack import partition_size

#: c = log2(e): 2^(c * d) == e^d, so the exp2 activation computes e^d.
LOG2E = 1.4426950408889634

#: One FP32 row.
ROW_BYTES = LANES * 4


def rvec_mask(plan) -> bytes:
    """softmax_rows_partial's rvec row (zeros), then its Pass-4 R_MASK image.

    The mask is 8 slots of 16 bytes (128 bits, bit 1 = keep), zero-padded to
    the row: slot p < P = 128 / ps keeps bits [p * ps, p * ps + n), so each
    partition's MULT.RC.VE (mask_offset = p) adds exactly its own row's
    elements and never a neighbour's.
    """
    n = plan.params["n"]
    slot = partition_size(n)
    mask = bytearray(LANES)
    for p in range(LANES // slot):
        bits = ((1 << n) - 1) << (p * slot)
        mask[p * 16 : (p + 1) * 16] = bits.to_bytes(16, byteorder="little")
    return bytes(ROW_BYTES) + bytes(mask) + bytes(ROW_BYTES - len(mask))


def log2e_keep(plan) -> bytes:
    """softmax_columns_packed's C_VEC row (log2(e) in every lane), then its keep
    row: 1.0 on each group's ``width`` real columns, 0.0 on its padding."""
    width = plan.params["width"]
    group = partition_size(width)
    keep = np.zeros((LANES // group, group), dtype="<f4")
    keep[:, :width] = 1.0
    return np.full(LANES, LOG2E, dtype="<f4").tobytes() + keep.tobytes()
