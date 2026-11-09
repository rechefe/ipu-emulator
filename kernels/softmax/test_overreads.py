"""The softmax kernels read no XMEM row past their data.

Pass 1 reads each row of ``x`` and Pass 2 reads it again; Pass 3 reads each row
of the numerators and Pass 4 reads them again. So a correct run reads every row
of ``x`` and of ``num`` exactly twice, and no row outside the placed regions.
The shapes below are the ones that once read an extra row: one row down a
column, one slot-packed vector, and a row length that 128 divides.

The read counts come from the reference model's XMEM read observer, not from
the kernel's own bookkeeping, so a wrong loop bound shows up as an extra read
whatever the data holds.
"""

from __future__ import annotations

from collections import Counter

import pytest

from ipu_harness.assembly import AssemblyCache
from ipu_harness.cases import at_values, expand
from ipu_harness.registry import default_registry
from ipu_harness.runner import RunOptions, run_kernel
from ipu_isa.formats import ElementFormat

CACHE = AssemblyCache()
ROW_BYTES = 512  # 128 FP32 lanes


@pytest.fixture(scope="module")
def registry():
    loaded = default_registry()
    assert not loaded.invalid, loaded.invalid
    return loaded


def xmem_reads(registry, kernel_name, **params):
    """Run ``kernel_name`` at ``params``; return the rows it read (a Counter by
    row), its regions and its cycles."""
    kernel = registry.kernel(kernel_name)
    default = next(c for c in expand(kernel) if c.name == "default")
    case = at_values(default, "overreads", {**params, "seed": 3, "scale": 3.0})
    reads: Counter = Counter()
    placed = {}

    def execute(context):
        placed.update(context.placement.regions)

        def note(address, size):
            first, last = address // ROW_BYTES, (address + size - 1) // ROW_BYTES
            reads.update(range(first, last + 1))

        context.model.observe_xmem_reads(note)
        context.model.run(context.cycle_limit)

    result = run_kernel(kernel, case, ElementFormat.FP32, RunOptions(executor=execute), cache=CACHE)
    assert result.passed, result.describe()
    return reads, placed, result.cycles


SHAPES = [
    # One row down a column, one slot-packed vector, and 128 dividing n.
    ("softmax_columns", {"rows": 1, "width": 200, "dim": 0}),
    ("softmax_columns", {"rows": 1, "width": 65, "dim": 0}),
    ("softmax_columns_packed", {"rows": 4, "width": 20, "dim": 0}),
    ("softmax_columns_packed", {"rows": 1, "width": 64, "dim": 0}),
    ("softmax_rows_long", {"rows": 3, "n": 256, "dim": 1}),
    ("softmax_rows_long", {"rows": 2, "n": 384, "dim": 1}),
    ("softmax_rows_long", {"rows": 1, "n": 128 * 8, "dim": 1}),
]


@pytest.mark.parametrize(("kernel", "params"), SHAPES, ids=lambda v: str(v))
def test_every_row_of_the_input_and_numerators_is_read_twice_and_none_past_the_data(
    registry, kernel, params
):
    reads, placed, _ = xmem_reads(registry, kernel, **params)
    inside = set()
    for region in placed.values():
        inside.update(range(region.base_row, region.base_row + region.rows))
    outside = sorted(set(reads) - inside)
    assert not outside, f"{kernel} {params}: read rows {outside} that no region holds"
    for name in ("x", "num"):
        region = placed[name]
        counts = [reads[r] for r in range(region.base_row, region.base_row + region.rows)]
        assert counts == [2] * region.rows, f"{kernel} {params}: reads of {name} by row: {counts}"
