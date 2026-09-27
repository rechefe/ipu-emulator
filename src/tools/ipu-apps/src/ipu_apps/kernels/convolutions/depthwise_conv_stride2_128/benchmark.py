"""Benchmark configs for depthwise_conv_stride2_128
(``bazel run :benchmark_depthwise_conv_stride2_128``): (rows, channels) sweep."""

CONFIGS = [
    dict(height=32, channels=8),     # small spatial, few channels
    dict(height=64, channels=16),    # multi-chunk, mid channels
    dict(height=128, channels=16),   # large spatial, primary benchmark
    dict(height=32, channels=64),    # many channels, small spatial
    dict(height=128, channels=32),   # dynamic region sizing at larger rows*channels
]
PER = "channels"
MAX_CYCLES = 50_000_000
