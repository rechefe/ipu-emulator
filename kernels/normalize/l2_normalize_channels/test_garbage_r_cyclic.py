"""The kernel does not read R_CYCLIC before it loads it.

A kernel starts with whatever the kernel before it left in the vector
registers. The first word of each column tile used to multiply R_CYCLIC by 0.0
to clear R_ACC; infinity or NaN in R_CYCLIC made that product NaN and every
sum with it. The first row of each tile now starts R_ACC itself.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from ipu_harness.chain import Chain
from ipu_isa.formats import ElementFormat
from ipu_isa.registers import REGISTER_DEFINITIONS

#: The default case and the two that end a tile or reduce a single row.
CASES = [(4, 16), (3, 133), (1, 129)]

POISON = {
    "nan": struct.pack("<f", float("nan")),
    "inf": struct.pack("<f", float("inf")),
    "minus_inf": struct.pack("<f", float("-inf")),
}


@pytest.mark.parametrize("poison", sorted(POISON))
@pytest.mark.parametrize(("channels", "columns"), CASES)
def test_a_poisoned_r_cyclic_does_not_change_the_output(channels, columns, poison):
    x = np.random.RandomState(7).normal(size=(channels, columns)).astype(np.float32)
    chain = Chain(ElementFormat.FP32)
    for bank in ("r_cyclic", "r_cyclic_wide_debug"):
        size = REGISTER_DEFINITIONS[bank]["size_bytes"]
        chain.model.write_register(bank, POISON[poison] * (size // 4))
    step = chain.run(
        "l2_normalize_channels", {"channels": channels, "columns": columns}, {"x": x}, check=True
    )
    assert step.result.passed, step.result.describe()
