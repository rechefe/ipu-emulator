"""softmax_rows_long: every case, plus the default case swept over sizes and scales."""
from ipu_apps.kernel_registry.testing import case_tests

test_case = case_tests(__package__, sweep=[
    dict(rows=1, n=129, scale=3.0, seed=0),       # smallest long row: 1 full chunk + 1 tail
    dict(rows=4, n=200, scale=4.0, seed=1),       # 1 full + 72 tail
    dict(rows=8, n=257, scale=3.0, seed=2),       # 2 full + 1 tail
    dict(rows=8, n=300, scale=5.0, seed=3),       # 2 full + 44 tail
    dict(rows=16, n=130, scale=50.0, seed=4),     # numerical stability (large |x|)
    dict(rows=8, n=300, scale=0.01, seed=5),      # near-uniform
    dict(rows=32, n=401, scale=3.0, seed=6),      # 3 full + 17 tail
    dict(rows=128, n=129, scale=4.0, seed=7),     # max rows in one group
    # n % 128 == 0: exactly full_chunks whole chunks, NO tail chunk. The tail
    # block still executes with valid_elements=0 (CR8), which makes its running
    # AGG.MAX/AGG.SUM exact no-ops -- so the same kernel covers this shape.
    dict(rows=6, n=256, scale=5.0, seed=8),       # 2 full chunks, no tail
    dict(rows=6, n=384, scale=5.0, seed=9),       # 3 full chunks, no tail
    dict(rows=4, n=512, scale=3.0, seed=10),      # 4 full chunks, no tail
    dict(rows=2, n=1024, scale=4.0, seed=11),     # 8 full chunks, no tail
    # >128 rows: maxvec/rvec hold one slot per row in a single 128-element vector,
    # so the kernel runs groups of <=128 rows (all four passes per group). Row
    # indices restart each group, which is what keeps them out of the R1 range
    # that MULT.RC.VE's `src` would otherwise select. See the .asm group loop.
    dict(rows=129, n=129, scale=3.0, seed=12),    # one row past a full group
    dict(rows=200, n=200, scale=4.0, seed=13),
    dict(rows=256, n=130, scale=3.0, seed=14),    # exactly two full groups
    dict(rows=300, n=129, scale=5.0, seed=15),    # two full groups + a short one
])
