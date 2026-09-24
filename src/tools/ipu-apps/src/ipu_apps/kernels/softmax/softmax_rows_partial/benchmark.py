"""Benchmark configs for softmax_rows_partial (``bazel run :benchmark_softmax_rows_partial``).

Covers every P (1/2/4/8) at low chunk counts, plus configs past 128 rows so the
per-group cost of the group loop stays measured (see STATUS.md).
"""

# The last three cross at least one 128-row group boundary, where the kernel
# re-runs all four passes on the next group -- included so that per-group
# overhead shows up in cyc/rows rather than hiding behind small inputs.
CONFIGS = [
    dict(n=8, rows=16), dict(n=16, rows=16), dict(n=32, rows=40),
    dict(n=64, rows=50), dict(n=128, rows=50),
    dict(n=16, rows=512), dict(n=64, rows=300), dict(n=128, rows=300),
]
PER = "rows"
