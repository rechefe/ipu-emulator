"""conv_universal: every case, plus a shape sweep (full/partial FPB=28
kernel blocks, cross-chunk spatial sizes, cols=128, padding-heavy shapes) and
its end-to-end layer shape (width 100, padded to cols=128)."""
from ipu_apps.kernel_registry.testing import case_tests

test_case = case_tests(__package__, sweep=[
    dict(in_channels=28, out_channels=4, height=16, width=16),   # exactly one full block
    dict(in_channels=56, out_channels=4, height=16, width=16),   # exactly two full blocks
    dict(in_channels=10, out_channels=4, height=16, width=16),   # single partial block
    dict(in_channels=16, out_channels=8, height=32, width=32),   # cross-chunk, multiple filters
    dict(in_channels=4, out_channels=2, height=8, width=128),    # cols=128 (Partition.P0)
    dict(in_channels=16, out_channels=4, height=8, width=128),   # cols=128, partial block
    dict(in_channels=4, out_channels=2, height=8, width=8),      # padding-heavy shapes
    dict(in_channels=4, out_channels=2, height=5, width=5),
    dict(in_channels=3, out_channels=4, height=8, width=100),    # non-power-of-2 width
])
