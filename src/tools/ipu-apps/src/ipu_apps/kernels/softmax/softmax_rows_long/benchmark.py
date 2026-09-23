"""Benchmark configs for softmax_rows_long (``bazel run :benchmark_softmax_rows_long``)."""

# n > 128, n % 128 != 0. The last two cross a 128-row group boundary, where the
# kernel re-runs all four passes on the next group -- kept here so the
# per-group overhead stays visible in cyc/rows.
CONFIGS = [
    dict(rows=8, n=200), dict(rows=8, n=300), dict(rows=16, n=500),
    dict(rows=4, n=1000), dict(rows=200, n=200), dict(rows=300, n=129),
]
PER = "rows"
