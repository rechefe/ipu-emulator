"""``kernel/<k>/*``: emulator speed and peak memory on two fixed kernel cases.

``identity`` (small) and ``matmul_720x240_x128`` (large: the largest standard
matmul case, the same kernel E04-02 benchmarks), each with its ``default``
case, run and checked exactly as the tests run them.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from tempfile import TemporaryDirectory

from ipu_bench.isolated import call, peak_rss_mb
from ipu_bench.result import Result, tiny_budget

SMALL = "identity"
LARGE = "matmul_720x240_x128"


class RunTimer:
    """A ``run_case`` recorder that only times ``app.run`` (the emulated cycles)."""

    seconds = 0.0

    @contextmanager
    def watch(self):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.seconds = time.perf_counter() - start

    def finish(self, state, cycles, output_path=None):
        pass


def measure(kernel: str) -> dict:
    """In a fresh interpreter: assemble, then time the run of the default case."""
    from ipu_apps.kernel_registry.cases import assemble_kernel, load_cases, run_case

    case = load_cases(kernel)["default"]
    timer = RunTimer()
    with TemporaryDirectory(prefix="ipu-bench-") as tmp:
        inst_path = assemble_kernel(kernel, tmp)
        _, cycles = run_case(kernel, case, inst_path=inst_path, recorder=timer)
    return {"cycles": cycles, "seconds": timer.seconds, "peak_rss_mb": peak_rss_mb()}


def run(quick: bool) -> list[Result]:
    results = []
    for kernel in (SMALL,) if tiny_budget() else (SMALL, LARGE):
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
