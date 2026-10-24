"""The score kernels' stored extent (E07-12): only the first n_tok lanes of a
stored row are scores.

There is no AGG in a score kernel, so lanes are independent: garbage in the
padding lanes of an input could only reach the padding lanes of the output.
What the DS (valid_elements = n_tok) controls is the stored extent:
ACTIVATE.EXACT writes the lanes past n_tok as zero when it stores the whole
row. Each kernel's ``padding_not_stored`` case puts 1e3 in its channel-major
input's padding lanes (``pack = "padded"``, ``pad = 1000.0``), as a chained
producer would leave another stream's data there; the raw stored rows must be
zero past n_tok, and the scores must still match their golden.
"""

from __future__ import annotations

import numpy as np
import pytest

from ipu_harness.cases import expand
from ipu_harness.registry import default_registry
from ipu_harness.runner import run_kernel
from ipu_isa.formats import ElementFormat
from ipu_isa.geometry import LANES


@pytest.mark.parametrize("name", ["qk_scores_16x60", "qk_scores_64x48", "attn_scores_km_16x60"])
def test_padding_lanes_are_not_stored(name):
    kernel = default_registry().kernel(name)
    case = next(c for c in expand(kernel) if c.name == "padding_not_stored")
    assert case.values["pad"] == 1000.0
    result = run_kernel(kernel, case, ElementFormat.FP32)
    assert result.passed, result.describe()
    rows = np.frombuffer(result.raw_outputs["y"], dtype="<f4").reshape(-1, LANES)
    n_tok = int(case.values["n_tok"])
    assert (rows[:, n_tok:] == 0.0).all(), "a padding lane was stored non-zero"
