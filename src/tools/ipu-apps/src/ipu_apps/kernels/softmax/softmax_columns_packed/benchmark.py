"""Benchmark configs for softmax_columns_packed (``bazel run :benchmark_softmax_columns_packed``)."""

# width in 1..64 (packed rows_per_vec = 128/width elements).
CONFIGS = [
    dict(rows=64, width=8), dict(rows=64, width=16),
    dict(rows=100, width=32), dict(rows=128, width=64),
]
PER = "rows"
