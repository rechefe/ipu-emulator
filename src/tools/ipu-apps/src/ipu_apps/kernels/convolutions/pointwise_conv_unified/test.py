"""pointwise_conv_unified: every case, plus configs -- one per code path
(single-pass, the 128-channel boundary, multi-pass with and without a tail,
padded non-power-of-2 and degenerate 1x1 spatial shapes) and its end-to-end
layer shape (160 channels at 8x8)."""
from ipu_apps.kernel_registry.testing import case_tests

test_case = case_tests(__package__, sweep=[
    dict(in_channels=128, out_channels=8, height=16, width=16),   # single pass, 128 boundary
    dict(in_channels=144, out_channels=8, height=16, width=16),   # multi-pass: 1 full + tail 16
    dict(in_channels=256, out_channels=8, height=16, width=16),   # multi-pass, no tail
    dict(in_channels=96, out_channels=32, height=32, width=32),   # larger spatial / out_ch
    dict(in_channels=144, out_channels=8, height=3, width=5),     # non-power-of-2 spatial
    dict(in_channels=8, out_channels=4, height=1, width=1),       # degenerate 1x1 spatial
    dict(in_channels=160, out_channels=160, height=8, width=8),
])
