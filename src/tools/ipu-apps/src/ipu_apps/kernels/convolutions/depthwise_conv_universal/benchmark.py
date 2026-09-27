"""Benchmark configs for depthwise_conv_universal
(``bazel run :benchmark_depthwise_conv_universal``)."""

CONFIGS = [
    dict(height=16, width=16, channels=8),    # partial single block
    dict(height=16, width=16, channels=28),   # exactly 1 full FPB=28 block
    dict(height=16, width=16, channels=29),   # 1 full + 1-channel partial
    dict(height=32, width=32, channels=16),   # multi-chunk, partial block
    dict(height=32, width=32, channels=32),   # multi-chunk, 1 full + partial
    dict(height=32, width=32, channels=56),   # exactly 2 full blocks
    dict(height=64, width=64, channels=32),   # primary benchmark
    dict(height=64, width=64, channels=64),   # large spatial, multiple blocks
    dict(height=32, width=32, channels=96),   # many channels
    dict(height=16, width=16, channels=40),   # two blocks, small spatial
]
PER = "channels"
MAX_CYCLES = 50_000_000
