"""Benchmark configs for maxpool2d_stride2_tail (``bazel run :benchmark_maxpool2d_stride2_tail``).

Widths whose last output XMEM row has one input XMEM row behind it (260, 640).
"""

CONFIGS = [
    dict(channels=1, height=2, width=260),
    dict(channels=64, height=2, width=260),
    dict(channels=1, height=480, width=640),
    dict(channels=64, height=480, width=640),
]
MAX_CYCLES = 20_000_000
