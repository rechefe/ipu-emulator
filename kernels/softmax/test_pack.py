"""The family's slot layouts (``ipu_kernels.softmax.pack``): what the ``ipu_apps``
harness wrote and read back, at the old family layout test's shapes."""

from __future__ import annotations

import numpy as np
import pytest

from ipu_kernels.softmax.pack import NEG_PAD, col_slots_negpad, partition_size, row_slots


@pytest.mark.parametrize(
    ("cols", "slot"), [(1, 16), (16, 16), (17, 32), (33, 64), (65, 128), (128, 128)]
)
def test_the_slot_is_the_next_power_of_two_from_16(cols, slot):
    assert partition_size(cols) == slot


@pytest.mark.parametrize("cols", [0, 129])
def test_a_slot_holds_1_to_128_elements(cols):
    with pytest.raises(ValueError, match="1..128"):
        partition_size(cols)


@pytest.mark.parametrize(
    ("pack", "fill"), [(row_slots, 0.0), (col_slots_negpad, NEG_PAD)], ids=["row", "col"]
)
@pytest.mark.parametrize("shape", [(10, 20), (8, 32), (7, 5), (1, 64), (33, 16)])
def test_rows_in_slots_round_trip_with_the_padding_filled(pack, fill, shape):
    rows, cols = shape
    a = np.arange(rows * cols, dtype=np.float64).reshape(shape) + 1
    stored = pack.to_rows(a, None)
    slot = partition_size(cols)
    per_row = 128 // slot
    assert stored.shape == (pack.rows(shape, None), 128) == (-(-rows // per_row), 128)
    slots = stored.reshape(-1, slot)
    np.testing.assert_array_equal(slots[:rows, :cols], a)  # row r in slot r
    assert (slots[:rows, cols:] == fill).all() and (slots[rows:] == fill).all()
    np.testing.assert_array_equal(pack.from_rows(stored, None, shape), a)
