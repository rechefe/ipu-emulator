"""``emulator/*``: the Q71 emulator targets (E04-02).

* ``IpuState()`` construction: time per construction, ``tracemalloc`` peak and
  the ``ru_maxrss`` growth of the first construction (targets: < 5 ms, < 5 MB).
* Cycles per second of a fixed matmul kernel run to ``HALT``, best of 5 (one
  run with ``--quick``; target: >= 50k cycles/s): ``MATMUL`` in FP32, the
  format it is written for, ``INT8`` (the ``default`` case of
  ``fully_connected`` in INT8; no matmul kernel is written for INT8) and
  ``E4`` (the same case in FP8 E4M3). Each is a kernel-tree case, run through
  the harness runner and checked. All run on the faithful path and, in a build
  with ``--define=ipu_proto=1``, also in prototyping mode; ``--tiny`` runs
  INT8 only.

``bazel run //src/tools/ipu-emu-py:bench_emulator`` runs only this module.
"""

from __future__ import annotations

import os

from ipu_bench.isolated import call, peak_rss_mb
from ipu_bench.result import Budget, Result

# The largest matmul whose default case runs in about 10 s on the faithful path
# (77185 cycles). matmul_720x240_x128 (the default in IMPLEMENTATION-DECISIONS.md,
# still kernels.LARGE) takes about 22 s, which breaks the 30 s bound on --quick.
#: (kernel, case, format, the result's name): the format None is the case's first.
MATMUL = ("matmul_384x192_x128", "default", None, "matmul_384x192_x128.default")
INT8 = ("fully_connected", "default", "int8", "fully_connected.int8")
# The E4 format (E04-17).
E4 = ("fully_connected", "default", "e4", "fully_connected.e4")
CONSTRUCT_NUMBER = 20


def construct() -> dict:
    """In a fresh interpreter: memory of the first ``IpuState()``, then its time."""
    import timeit
    import tracemalloc

    from ipu_emu.ipu_state import IpuState

    rss_before = peak_rss_mb()
    tracemalloc.start()
    state = IpuState()
    _, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_delta = peak_rss_mb() - rss_before
    del state
    seconds = timeit.timeit(IpuState, number=CONSTRUCT_NUMBER) / CONSTRUCT_NUMBER
    return {"seconds": seconds, "traced_mb": traced_peak / (1024 * 1024), "rss_mb": rss_delta}


def cycles(kernel: str, case_name: str, fmt: str | None, repeats: int, mode: str) -> dict:
    """In a fresh interpreter: assemble once, run the case ``repeats`` times.

    ``mode`` is ``"faithful"`` or ``"prototyping"``.

    Only the emulated run is timed; each run is checked as the tests check it.
    """
    # Read when each IpuState is built; set on or off explicitly, because a
    # --define=ipu_proto=1 build makes prototyping the default.
    os.environ["IPU_EMU_PROTOTYPING"] = "1" if mode == "prototyping" else "0"
    from ipu_bench.kernel_runs import run_timed

    return run_timed(kernel, case_name, fmt, repeats=repeats)


def _prototyping_built() -> bool:
    from ipu_emu.prototyping import build_enabled

    return build_enabled()


def run(budget: Budget) -> list[Result]:
    c = call("ipu_bench.benchmarks.emulator:construct")
    first = "the first IpuState() in a fresh interpreter"
    results = [
        Result(
            "emulator/ipu_state_construct_ms",
            "ms",
            c["seconds"] * 1000,
            extra=f"IpuState(), mean of {CONSTRUCT_NUMBER}",
        ),
        Result("emulator/ipu_state_tracemalloc_peak_mb", "MB", c["traced_mb"], extra=first),
        # ru_maxrss is a high-water mark: a construction below the existing peak reads 0.
        Result("emulator/ipu_state_rss_delta_mb", "MB", c["rss_mb"], extra=first),
    ]
    # The smoke test runs the INT8 case (2674 cycles) once per mode, nothing larger.
    repeats = 5 if budget is Budget.FULL else 1
    kernels = (INT8,) if budget is Budget.TINY else (MATMUL, INT8, E4)
    modes = ("faithful", "prototyping") if _prototyping_built() else ("faithful",)
    for mode in modes:
        for kernel, case_name, fmt, label in kernels:
            m = call(
                "ipu_bench.benchmarks.emulator:cycles",
                kernel=kernel,
                case_name=case_name,
                fmt=fmt,
                repeats=repeats,
                mode=mode,
            )
            results.append(
                Result(
                    f"emulator/{mode}/{label}/cycles_per_s",
                    "cycles/s",
                    m["cycles"] / m["seconds"],
                    extra=f"cycles={m['cycles']} best of {repeats}: {m['seconds']:.3f} s",
                    bigger_is_better=True,
                )
            )
    return results
