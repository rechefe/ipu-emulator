"""Benchmark config for conv_first_layer (``bazel run :benchmark_conv_first_layer``).

Fixed shape (256x256x3 -> 128x128x16, stride 2), so a single config: the
default case.
"""

CONFIGS = [dict(seed=42)]
MAX_CYCLES = 50_000_000
