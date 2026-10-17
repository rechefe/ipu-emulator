"""The family's pack transform (``ipu_kernels.attention.pack``): ``padded`` is
the default row-major layout with the padding lanes set to ``pad``."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ipu_harness.packing import DEFAULT
from ipu_kernels.attention.pack import padded


@pytest.mark.parametrize("shape", [(3, 16), (2, 5, 64), (4, 256), (2, 200)])
def test_padded_is_the_default_layout_with_the_padding_set(shape):
    a = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) + 1
    zero = SimpleNamespace(params={"pad": 0.0})
    np.testing.assert_array_equal(padded.to_rows(a, zero), DEFAULT.to_rows(a, zero))
    garbage = padded.to_rows(a, SimpleNamespace(params={"pad": 1e3}))
    is_padding = DEFAULT.to_rows(np.zeros(shape), zero) != DEFAULT.to_rows(np.ones(shape), zero)
    assert (garbage[~is_padding] == 1e3).all()
    np.testing.assert_array_equal(garbage[is_padding], DEFAULT.to_rows(a, zero)[is_padding])
    np.testing.assert_array_equal(padded.from_rows(garbage, zero, shape), a)
