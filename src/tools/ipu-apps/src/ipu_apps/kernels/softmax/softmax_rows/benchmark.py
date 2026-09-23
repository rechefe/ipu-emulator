"""Benchmark configs for softmax_rows (``bazel run :benchmark_softmax_rows``)."""

CONFIGS = [dict(rows=rows) for rows in (8, 32, 128, 256, 500)]
PER = "rows"
