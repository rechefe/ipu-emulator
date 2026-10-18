"""softmax_rows_partial's multiplier counts: every partition is multiplied,
its shifted Pass-4 write included (the stats_* cases).

The old test also counted identity-multiply cycles (2 per row); the
reference model's statistics count identity lanes, not cycles, and the lane
count below pins the same multiplies."""

from __future__ import annotations

import pytest

from ipu_isa.formats import ElementFormat


@pytest.mark.parametrize(("n", "rows"), [(16, 8), (32, 8), (50, 4), (100, 2)])
def test_every_partition_is_multiplied(cases, run_case, n, rows):
    (case,) = [c for c in cases if c.name == f"stats_n{n}"]
    assert (case.values["n"], case.values["rows"]) == (n, rows)
    stats = run_case(case, ElementFormat.FP32).stats
    assert stats is not None
    assert stats.mult_lane_ops == 5 * rows * n  # five multiplies per row
    assert stats.mult_identity_lane_ops == 2 * rows * n  # two by 1.0 per row
