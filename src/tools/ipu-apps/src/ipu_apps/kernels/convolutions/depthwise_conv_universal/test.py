"""depthwise_conv_universal: every case, plus a shape sweep (full and
partial FPB=28 super-blocks, multi-chunk, cols=128, padding-heavy shapes) and
its end-to-end layer shape (width 128, 5 rows)."""
from ipu_apps.kernel_registry.testing import case_tests

test_case = case_tests(__package__, sweep=[
    dict(channels=28, height=16, width=16),   # exactly one full FPB=28 super-block
    dict(channels=32, height=16, width=16),   # one full + one partial super-block
    dict(channels=16, height=32, width=32),   # multi-chunk, partial super-block
    dict(channels=50, height=64, width=64),   # two super-blocks, larger spatial
    dict(channels=4, height=8, width=128),    # cols=128 (Partition.P0)
    dict(channels=30, height=16, width=128),  # cols=128, multi-chunk + super-block spanning
    dict(channels=4, height=8, width=8),      # padding-heavy shapes
    dict(channels=4, height=5, width=5),
    dict(channels=3, height=5, width=128),
])
