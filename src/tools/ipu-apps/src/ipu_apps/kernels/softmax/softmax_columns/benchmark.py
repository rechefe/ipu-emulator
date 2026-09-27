"""Benchmark configs for softmax_columns (``bazel run :benchmark_softmax_columns``)."""

# width >= 128, padded up to the next multiple of 128 internally.
CONFIGS = [
    dict(rows=16, width=128), dict(rows=64, width=128), dict(rows=128, width=128),
    dict(rows=32, width=256), dict(rows=64, width=384),
]
PER = "rows"
