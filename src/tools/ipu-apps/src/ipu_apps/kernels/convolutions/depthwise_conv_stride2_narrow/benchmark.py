"""Benchmark configs for depthwise_conv_stride2_narrow
(``bazel run :benchmark_depthwise_conv_stride2_narrow``)."""

CONFIGS = [
    dict(height=8, width=64, channels=2),     # cols=64, minimal
    dict(height=8, width=64, channels=16),    # cols=64, more channels
    dict(height=16, width=32, channels=4),    # cols=32, small
    dict(height=16, width=32, channels=24),   # cols=32, larger channel count
    dict(height=32, width=16, channels=3),    # cols=16, odd channel count
    dict(height=32, width=16, channels=16),   # cols=16, larger channel count
]
PER = "channels"
MAX_CYCLES = 50_000_000
