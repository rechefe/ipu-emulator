"""Run one measurement in a fresh interpreter, so imports and memory are not shared.

``call("ipu_bench.benchmarks.kernels:measure", kernel="identity")`` starts
``python -m ipu_bench.isolated``, which imports the function, calls it with the
keyword arguments and prints its JSON result on the last line of stdout.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from importlib import import_module

_MARKER = "IPU_BENCH_RESULT "


def child_env() -> dict[str, str]:
    """This interpreter's ``sys.path`` (Bazel runfiles included) for a child."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    return env


def call(target: str, **kwargs):
    """``target`` is ``module:function``; returns the function's JSON-able result."""
    proc = subprocess.run(
        [sys.executable, "-m", "ipu_bench.isolated", target, json.dumps(kwargs)],
        env=child_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    lines = [line for line in proc.stdout.splitlines() if line.startswith(_MARKER)]
    if proc.returncode != 0 or not lines:
        raise RuntimeError(
            f"{target}({kwargs}) failed with exit code {proc.returncode}:\n"
            f"{proc.stdout[-2000:]}{proc.stderr[-4000:]}"
        )
    return json.loads(lines[-1][len(_MARKER) :])


def python_wall_s(code: str) -> float:
    """Wall time of ``python -c <code>`` in a fresh interpreter, startup included."""
    start = time.perf_counter()
    subprocess.run([sys.executable, "-c", code], env=child_env(), check=True)
    return time.perf_counter() - start


def peak_rss_mb() -> float:
    """This process's peak resident set size (Linux reports KiB, macOS bytes)."""
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def main(argv: list[str]) -> int:
    target, kwargs = argv
    module_name, _, function_name = target.partition(":")
    result = getattr(import_module(module_name), function_name)(**json.loads(kwargs))
    print(_MARKER + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
