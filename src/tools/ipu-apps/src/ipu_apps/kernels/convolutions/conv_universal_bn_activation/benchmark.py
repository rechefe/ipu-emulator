"""Benchmark configs for conv_universal_bn_activation
(``bazel run :benchmark_conv_universal_bn_activation``)."""

CONFIGS = [
    dict(height=16, width=16, in_channels=28, out_channels=4),   # exactly one full super-block
    dict(height=16, width=16, in_channels=16, out_channels=4),   # partial last block
    dict(height=32, width=32, in_channels=28, out_channels=8),   # multi-chunk
    dict(height=32, width=32, in_channels=32, out_channels=16),  # two super-blocks, multi-filter
    dict(height=32, width=32, in_channels=16, out_channels=32),  # more filters than channels
    dict(height=64, width=64, in_channels=32, out_channels=32),  # primary benchmark
    dict(height=64, width=64, in_channels=28, out_channels=14),  # one full super-block, larger spatial
    dict(height=32, width=32, in_channels=64, out_channels=8),   # large in_ch
    dict(height=16, width=16, in_channels=56, out_channels=4),   # two super-blocks (bias once)
]
MAX_CYCLES = 50_000_000
