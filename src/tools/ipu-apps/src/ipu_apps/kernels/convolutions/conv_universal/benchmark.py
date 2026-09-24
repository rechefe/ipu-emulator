"""Benchmark configs for conv_universal (``bazel run :benchmark_conv_universal``).

Configs: (height, width, in_channels, out_channels) around the
FPB=28 super-block boundaries, up to the 64x64x32->32 primary benchmark.
"""

CONFIGS = [
    dict(height=16, width=16, in_channels=28, out_channels=4),   # exactly 1 full block
    dict(height=16, width=16, in_channels=16, out_channels=4),   # partial last block
    dict(height=16, width=16, in_channels=56, out_channels=4),   # exactly 2 full blocks
    dict(height=32, width=32, in_channels=28, out_channels=8),   # 1 full block, multi-chunk
    dict(height=32, width=32, in_channels=32, out_channels=16),  # partial last block, multi-filter
    dict(height=32, width=32, in_channels=16, out_channels=32),  # more filters than channels
    dict(height=64, width=64, in_channels=32, out_channels=32),  # primary benchmark
    dict(height=64, width=64, in_channels=28, out_channels=28),  # 1 full block, larger spatial
    dict(height=32, width=32, in_channels=64, out_channels=8),   # large in_ch
    dict(height=16, width=16, in_channels=84, out_channels=4),   # exactly 3 full blocks
]
MAX_CYCLES = 50_000_000
