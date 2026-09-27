"""Benchmark configs for pointwise_conv_unified (``bazel run :benchmark_pointwise_conv_unified``)."""

CONFIGS = [
    dict(height=128, width=128, in_channels=64, out_channels=32),
    dict(height=128, width=128, in_channels=32, out_channels=48),
    dict(height=64, width=64, in_channels=128, out_channels=64),
    dict(height=64, width=64, in_channels=64, out_channels=128),
    dict(height=32, width=32, in_channels=256, out_channels=96),
    dict(height=32, width=32, in_channels=96, out_channels=256),
    dict(height=16, width=16, in_channels=384, out_channels=128),
    dict(height=16, width=16, in_channels=128, out_channels=256),
]
MAX_CYCLES = 200_000_000
