"""The key-major score kernels' key loop runs once per key, no more.

The loop counter's ``ADD`` once sat in the same word as the ``BLT.S``, which
reads the cycle-start value, so the loop ran one key too many: it stored an
extra score row (one per query group) past the scores. These kernels hold no
other store, so the model's count of XMEM rows written is exactly the score
rows: ``n_tok`` keys times one row per 128 queries.
"""

from __future__ import annotations

import pytest

from ipu_harness.cases import expand
from ipu_harness.registry import default_registry
from ipu_harness.runner import run_kernel
from ipu_isa.formats import ElementFormat
from ipu_isa.geometry import LANES


@pytest.mark.parametrize("name", ["attn_scores_km_16x60", "attn_scores_km_256x36"])
def test_one_score_row_stored_per_key_and_group(name):
    kernel = default_registry().kernel(name)
    case = next(c for c in expand(kernel) if c.name == "default")
    result = run_kernel(kernel, case, ElementFormat.FP32)
    assert result.passed, result.describe()
    n_tok = int(case.values["n_tok"])
    groups = -(-n_tok // LANES)
    assert result.stats.xmem_writes == n_tok * groups
