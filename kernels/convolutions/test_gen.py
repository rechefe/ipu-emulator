"""The legacy case data (``gen.py``, E07-16): the tiled kernels' x, w and b
come from one generator in turn, so they share no values and a kernel that
reads the wrong region (b where it should read x) cannot pass."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location("conv_gen", Path(__file__).with_name("gen.py"))
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("shape", [(3, 3, 8, 2), (1, 1, 1, 1)])
def test_tiled_case_tensors_share_no_values(k, shape):
    c, h, w, o = shape
    plan = SimpleNamespace(
        params=dict(in_channels=c, height=h, width=w, out_channels=o, kernel_size=k)
    )
    made = gen.legacy_tiled(plan, None)
    x, weights, bias = (made[n].ravel() for n in ("x", "w", "b"))
    first = np.random.default_rng(7).standard_normal(x.size).astype("<f4")
    assert x.tobytes() == first.tobytes()  # x is the stream's first draws
    assert not np.isin(bias, x).any()
    assert not np.isin(weights * 4, x).any()
    assert not np.isin(bias, weights * 4).any()
