"""softmax_columns: every case, plus the default case swept over sizes and scales."""
from ipu_apps.kernel_registry.testing import case_tests

test_case = case_tests(__package__, sweep=[
    dict(rows=8, width=128, scale=3.0, seed=0),      # single full chunk, few rows
    dict(rows=64, width=128, scale=4.0, seed=1),     # square 64x128
    dict(rows=1, width=128, scale=5.0, seed=2),      # single row -> softmax is all 1.0
    dict(rows=16, width=130, scale=3.0, seed=3),     # width 130 -> padded to 256 (2 chunks)
    dict(rows=32, width=200, scale=4.0, seed=4),     # width 200 -> padded to 256
    dict(rows=128, width=256, scale=3.0, seed=5),    # full 2-chunk width, many rows
    dict(rows=64, width=192, scale=50.0, seed=6),    # large |x| (stability), 192 -> 256
    dict(rows=10, width=129, scale=0.01, seed=7),    # near-uniform, 129 -> 256
    dict(rows=256, width=256, scale=3.0, seed=8),    # 256 rows (no row-group cap)
    dict(rows=32, width=300, scale=4.0, seed=9),     # width 300 -> padded to 384 (3 chunks)
    dict(rows=16, width=460, scale=3.0, seed=10),    # width 460 -> padded to 512 (4 chunks)
    dict(rows=8, width=384, scale=50.0, seed=11),    # exact 3-chunk width, large |x|
    # Sub-128 widths (65..127): one chunk, mostly padding. Correct because each
    # element is an INDEPENDENT column -- padding elements are their own (all-zero)
    # columns and never enter a real column's reduce. Widths <= 64 belong to
    # softmax_columns_packed, which fits several whole rows per vector.
    dict(rows=32, width=65, scale=4.0, seed=12),     # narrowest supported width
    dict(rows=32, width=96, scale=3.0, seed=13),
    dict(rows=32, width=127, scale=5.0, seed=14),    # widest sub-128 width
    dict(rows=16, width=100, scale=50.0, seed=15),   # sub-128 + large |x| (stability)
])
