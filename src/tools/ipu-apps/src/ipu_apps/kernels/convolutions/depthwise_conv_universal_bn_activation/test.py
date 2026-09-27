"""depthwise_conv_universal_bn_activation: every case, plus a shape
sweep (full and partial FPB=25 super-blocks, multi-chunk, cols=128,
padding-heavy shapes) and its end-to-end layer shape (width 65)."""
from ipu_apps.kernel_registry.testing import case_tests

test_case = case_tests(__package__, sweep=[
    dict(channels=25, height=16, width=16),   # exactly one full FPB=25 super-block
    dict(channels=32, height=16, width=16),   # one full + one partial super-block
    dict(channels=16, height=32, width=32),   # multi-chunk, partial super-block
    dict(channels=50, height=64, width=64),   # two super-blocks, larger spatial
    dict(channels=4, height=8, width=128),    # cols=128 (Partition.P0)
    dict(channels=30, height=16, width=128),  # cols=128, multi-chunk + super-block spanning
    dict(channels=4, height=8, width=8),      # padding-heavy shapes
    dict(channels=4, height=5, width=5),
    dict(channels=4, height=6, width=65),
])
