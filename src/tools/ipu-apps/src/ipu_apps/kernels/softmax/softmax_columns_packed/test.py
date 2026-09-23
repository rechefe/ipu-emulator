"""softmax_columns_packed: every case, plus the default case swept over sizes and scales."""
from ipu_apps.kernel_registry.testing import case_tests

test_case = case_tests(__package__, sweep=[
    dict(rows=16, width=64, scale=3.0, seed=0),     # 2 rows/vec, clean width
    dict(rows=64, width=32, scale=4.0, seed=1),     # 4 rows/vec
    dict(rows=100, width=16, scale=3.0, seed=2),    # 8 rows/vec, rows not a multiple of rpv (tail)
    dict(rows=33, width=16, scale=5.0, seed=3),     # 8 rows/vec, 33 rows -> 5 vectors (tail of 1)
    dict(rows=16, width=33, scale=3.0, seed=4),     # width 33 -> pad to 64 (intra-group padding)
    dict(rows=32, width=20, scale=4.0, seed=5),     # width 20 -> pad to 32
    dict(rows=16, width=15, scale=50.0, seed=6),    # width 15 -> pad to 16, large |x| (stability)
    dict(rows=8, width=10, scale=0.01, seed=7),     # width 10 -> pad to 16, near-uniform
    dict(rows=1, width=64, scale=3.0, seed=8),      # single row -> softmax all 1.0
])
