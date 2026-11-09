"""softmax_rows_partial's multiplier counts: every partition is multiplied,
its shifted Pass-4 write included (the stats_* cases).

The old test also counted identity-multiply cycles (2 per row); the
reference model's statistics count identity lanes, not cycles, and the lane
count below pins the same multiplies."""

from __future__ import annotations

import pytest

from ipu_harness.cases import at_values
from ipu_harness.runner import RunOptions
from ipu_isa.formats import ElementFormat


@pytest.mark.parametrize(("n", "rows"), [(16, 8), (32, 8), (50, 4), (100, 2)])
def test_every_partition_is_multiplied(cases, run_case, n, rows):
    (case,) = [c for c in cases if c.name == f"stats_n{n}"]
    assert (case.values["n"], case.values["rows"]) == (n, rows)
    stats = run_case(case, ElementFormat.FP32).stats
    assert stats is not None
    assert stats.mult_lane_ops == 5 * rows * n  # five multiplies per row
    assert stats.mult_identity_lane_ops == 2 * rows * n  # two by 1.0 per row


@pytest.mark.parametrize(("n", "rows"), [(8, 24), (8, 64), (16, 40), (32, 36), (64, 70)])
def test_every_row_the_program_touches_lies_inside_num(cases, run_case, n, rows):
    """The numerators are indexed by token slot: every slot of a group must fall in ``num``.

    A group has up to min(device_rows * parts, 128) token slots, one ``num`` row each;
    at n = 8, rows = 64 that is 64 rows against ``device_rows`` = 4.
    """
    (default,) = [c for c in cases if c.name == "default"]
    case = at_values(
        default, f"num_rows[n={n},rows={rows}]", {**default.values, "n": n, "rows": rows}
    )
    result = run_case(case, ElementFormat.FP32, RunOptions(tables=True))
    assert result.passed, result.describe()
    assert result.tables is not None
    num = result.tables.profiles["num"]
    assert num.accesses, "the program touches num"
    assert [a.describe() for a in num.outside] == []
