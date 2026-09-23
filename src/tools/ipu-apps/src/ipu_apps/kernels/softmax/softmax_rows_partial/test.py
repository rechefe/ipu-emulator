"""softmax_rows_partial: every case, plus the default case swept over P=1/2/4/8.

Includes the high chunk counts that used to trigger Pass-4 cross-partition
contamination (see STATUS.md) before the per-partition R_MASK fix, and row
counts past 128, which used to overflow MULT.RC.VE's src operand before the
group loop. The case check reads the app's OUTPUT FILE rather than XMEM, so it
also pins the round-trip property: output layout == input layout (row-major
rows x N).
"""
import pytest

from ipu_apps.kernel_registry.cases import load_cases, run_case
from ipu_apps.kernel_registry.testing import case_tests

test_case = case_tests(__package__, sweep=[
    # P=1 (N 65..128)
    dict(n=128, rows=4, seed=0, scale=3.0),
    dict(n=100, rows=6, seed=1, scale=3.0),
    dict(n=65, rows=3, seed=2, scale=3.0),
    # P=2 (N 33..64)
    dict(n=64, rows=4, seed=3, scale=3.0),
    dict(n=50, rows=10, seed=4, scale=3.0),
    dict(n=33, rows=8, seed=5, scale=3.0),
    # P=4 (N 17..32), incl. multi-chunk
    dict(n=32, rows=8, seed=6, scale=3.0),
    dict(n=32, rows=12, seed=7, scale=3.0),
    dict(n=20, rows=12, seed=8, scale=3.0),
    dict(n=17, rows=16, seed=9, scale=3.0),
    # P=8 (N 1..16), up to 2 chunks
    dict(n=16, rows=8, seed=10, scale=3.0),
    dict(n=16, rows=16, seed=11, scale=3.0),
    dict(n=8, rows=8, seed=12, scale=3.0),
    dict(n=8, rows=16, seed=13, scale=3.0),
    dict(n=1, rows=8, seed=14, scale=3.0),
    # Rows not a multiple of P are zero-padded internally (P=4, padded to 8).
    dict(n=32, rows=6, seed=20, scale=3.0),
    # High chunk counts that used to trigger Pass-4 cross-partition
    # contamination: each partition's MULT.RC.VE produces a full 128-lane result
    # where only [p*ps, p*ps+N) is that partition's own data -- the rest is real
    # (not zero) data from OTHER rows via the rc_idx wraparound trick.
    # acc.add/acc.add.first used to write/accumulate all 128 lanes unmasked, so
    # this leaked data corrupted neighboring partitions' slots once chunk count
    # was high enough for the leaked values to exceed tolerance. Fixed via a
    # per-partition R_MASK slot (mask_offset=p) that restricts each MULT.RC.VE
    # to exactly its own lane range -- see _partition_masks() and the .asm's
    # p4_run_p* blocks.
    dict(n=64, rows=70, seed=0, scale=5.0),     # P=2, num_chunks=35 -- broke before the fix
    dict(n=32, rows=36, seed=0, scale=5.0),     # P=4, num_chunks=9  -- broke before the fix
    dict(n=8, rows=24, seed=0, scale=5.0),      # P=8, num_chunks=3  -- the originally-documented case
    dict(n=8, rows=64, seed=0, scale=5.0),      # P=8, num_chunks=8
    dict(n=16, rows=40, seed=0, scale=5.0),     # P=8 (n=16), num_chunks=5
    # Rows beyond 128 used to silently corrupt: lr_row feeds MULT.RC.VE's `src`
    # scalar-select and AGG's dest slot, both of which index R0 for 0..127 and
    # switch to the never-loaded R1 at 128 (see STATUS.md's row-count
    # overflow). The kernel now processes groups of at most 128/P chunks
    # (= 128 logical rows), restarting lr_row each group, so the index can't
    # reach 128. These configs all cross at least one group boundary.
    dict(n=128, rows=129, seed=129, scale=3.0),    # P=1: one row past a full group
    dict(n=128, rows=300, seed=300, scale=3.0),    # P=1: two full groups + a short one
    dict(n=100, rows=257, seed=257, scale=3.0),    # P=1, N<ps
    dict(n=64, rows=130, seed=130, scale=3.0),     # P=2: one row past a full group
    dict(n=64, rows=1000, seed=1000, scale=3.0),   # P=2, many groups
    dict(n=32, rows=300, seed=300, scale=3.0),     # P=4
    dict(n=16, rows=500, seed=500, scale=3.0),     # P=8
    dict(n=8, rows=1032, seed=1032, scale=3.0),    # P=8: exactly 129 chunks -> group boundary + short group
    dict(n=20, rows=2000, seed=2000, scale=3.0),   # P=4, many groups
    dict(n=1, rows=600, seed=600, scale=3.0),      # P=8, degenerate N=1 (every row softmaxes to 1.0)
])


@pytest.mark.parametrize("n,rows", [(16, 8), (32, 8), (50, 4), (100, 2)])
def test_scatter_counts_every_partition(n, rows):
    state, _ = run_case("softmax_rows_partial", load_cases("softmax_rows_partial")["default"],
                        options={"n": n, "rows": rows})
    # Five multiplies per row, including each partition's shifted final write.
    assert state.stats.mult_lane_ops == 5 * rows * n
    assert state.stats.mult_identity_lane_ops == 2 * rows * n
    assert state.stats.mult_identity_cycles == 2 * rows
