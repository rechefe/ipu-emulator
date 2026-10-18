"""The softmax family's slot-packed layouts (E07-11).

``kernel.toml`` names them (``pack = "row_slots"``); the harness finds them in
``ipu_kernels.softmax.pack``.

``softmax_rows_partial`` and ``softmax_columns_packed`` give each logical row
of a ``(rows, cols)`` tensor a slot of ``partition_size(cols)`` lanes (the next
power of two in 16..128), so ``128 / slot`` rows share one XMEM row. The last
XMEM row is filled up with whole padding slots, and the lanes past ``cols`` in
each slot are padding too, as the old harness laid them out
(``SoftmaxApp.setup``). On unpack the padding is dropped.

``row_slots`` pads with 0.0 (the row kernel masks the padding off);
``col_slots_negpad`` pads with ``NEG_PAD`` = -1e30, which loses every column
max and underflows to 0 in every sum, so the column kernel's cross-group fold
ignores it. ``NEG_PAD`` is an FP32 value: the kernels run in FP32 only.
"""

from __future__ import annotations

import numpy as np

from ipu_harness.params import CasePlan
from ipu_isa.geometry import LANES

#: The column kernel's padding value.
NEG_PAD = -1.0e30


def partition_size(cols: int) -> int:
    """The slot width of a ``cols``-wide row: the next power of two >= cols, in 16..128.

    Raises:
        ValueError: ``cols`` is not in 1..128.
    """
    if not 1 <= cols <= LANES:
        raise ValueError(f"a slot holds 1..{LANES} elements, not {cols}")
    slot = 16
    while slot < cols:
        slot *= 2
    return slot


class Slots:
    """Each row of a ``(rows, cols)`` tensor in its own slot of lanes.

    A slot is ``partition_size(cols)`` lanes, ``128 / slot`` slots per XMEM
    row; padding lanes and padding slots hold ``fill``. Values are float64,
    as the default transform's; the harness encodes them in the format.

    Args:
        fill: The padding value.
    """

    def __init__(self, fill: float) -> None:
        self.fill = fill

    def rows(self, shape: tuple[int, ...], plan: CasePlan) -> int:
        """The XMEM rows: ``ceil(rows * slot / 128)``.

        Raises:
            ValueError: The tensor is not 2-D, or a row is wider than 128.
        """
        if len(shape) != 2:
            raise ValueError(f"slots take a (rows, cols) tensor, not shape {shape}")
        rows, cols = shape
        return -(-rows * partition_size(cols) // LANES)

    def to_rows(self, a: np.ndarray, plan: CasePlan) -> np.ndarray:
        """``a`` as ``(rows, 128)``: row r in lanes ``[r % P * slot, ... + cols)``
        of XMEM row ``r // P``, everything else ``fill``."""
        rows, cols = a.shape
        slot = partition_size(cols)
        device = np.full((self.rows(a.shape, plan) * LANES // slot, slot), self.fill)
        device[:rows, :cols] = a
        return device.reshape(-1, LANES)

    def from_rows(self, r: np.ndarray, plan: CasePlan, shape: tuple[int, ...]) -> np.ndarray:
        """The ``(rows, cols)`` tensor ``r`` holds (inverse of :meth:`to_rows`)."""
        rows, cols = shape
        return r.reshape(-1, partition_size(cols))[:rows, :cols]


#: softmax_rows_partial's x and y.
row_slots = Slots(0.0)

#: softmax_columns_packed's x and y.
col_slots_negpad = Slots(NEG_PAD)
