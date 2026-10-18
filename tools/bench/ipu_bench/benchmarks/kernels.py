"""``kernel/<k>/*``: emulator speed and peak memory on two fixed kernel cases.

``identity`` (small) and ``matmul_720x240_x128`` (large: the largest standard
matmul case; E04-02's ``emulator`` module uses a smaller one), each with its
``default`` case, run through the harness runner and checked exactly as
``bazel test //kernels/...`` runs them.
"""

from __future__ import annotations

from ipu_bench.isolated import call, peak_rss_mb
from ipu_bench.result import Budget, Result

SMALL = "identity"
LARGE = "matmul_720x240_x128"


def measure(kernel: str) -> dict:
    """In a fresh interpreter: assemble, then time the run of the default case."""
    from ipu_bench.kernel_runs import run_timed

    m = run_timed(kernel, "default")
    return {**m, "peak_rss_mb": peak_rss_mb()}


def run(budget: Budget) -> list[Result]:
    results = []
    for kernel in (SMALL,) if budget is Budget.TINY else (SMALL, LARGE):
        m = call("ipu_bench.benchmarks.kernels:measure", kernel=kernel)
        extra = f"default case; cycles={m['cycles']} run={m['seconds']:.3f} s"
        results += [
            Result(
                f"kernel/{kernel}/cycles_per_s",
                "cycles/s",
                m["cycles"] / m["seconds"],
                extra=extra,
                bigger_is_better=True,
            ),
            Result(f"kernel/{kernel}/peak_rss_mb", "MB", m["peak_rss_mb"], extra=extra),
        ]
    return results
