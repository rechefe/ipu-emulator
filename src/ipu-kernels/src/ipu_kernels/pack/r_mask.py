"""R_MASK rows: the bytes ``LD.MASK`` reads, for a scratch region's ``bytes``.

R_MASK is 1024 bits, 8 slots of 128, one bit per lane; ``LD.MASK`` reads them
from the first 128 bytes of a row, bit ``i`` in byte ``i // 8``, least
significant first. No element value can spell these bytes (some are NaN
patterns in FP32), so a mask is a scratch region whose starting bytes a
family's ``gen.py`` returns from these helpers (E07-23).
"""

from __future__ import annotations

import numpy as np

from ipu_isa.geometry import LANES

#: R_MASK's bits: 8 slots of one bit per lane.
MASK_BITS = 8 * LANES


def _row(bits: np.ndarray, row_bytes: int) -> bytes:
    """``bits`` as the first bytes of one row of ``row_bytes`` bytes."""
    raw = np.packbits(bits.astype(bool), bitorder="little").tobytes()
    return raw + bytes(row_bytes - len(raw))


def window_mask(tokens: int, row_bytes: int) -> bytes:
    """The packed layout's window mask: ``128 // tokens`` channels share a row,
    and slot ``p`` enables lanes ``[tokens * p, tokens * p + tokens)``, the
    window of the row's channel ``p``. The other slots are zero.

    Raises:
        ValueError: ``tokens`` does not divide 128.
    """
    if tokens < 1 or LANES % tokens:
        raise ValueError(f"window_mask: tokens ({tokens}) must divide {LANES}")
    bits = np.zeros(MASK_BITS)
    for p in range(LANES // tokens):
        start = LANES * p + tokens * p
        bits[start : start + tokens] = 1
    return _row(bits, row_bytes)


def all_ones(row_bytes: int) -> bytes:
    """Every R_MASK bit set, as one row: R_MASK's reset value."""
    return _row(np.ones(MASK_BITS), row_bytes)
