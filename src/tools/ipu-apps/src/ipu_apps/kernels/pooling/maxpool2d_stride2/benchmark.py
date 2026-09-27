"""Benchmark configs for maxpool2d_stride2 (``bazel run :benchmark_maxpool2d_stride2``)."""

CONFIGS = [
    dict(channels=1, height=2, width=256),
    dict(channels=64, height=2, width=256),
]
