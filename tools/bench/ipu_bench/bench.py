"""Run every benchmark module and write ``bench-smaller.json`` / ``bench-bigger.json``.

    bazel run //tools/bench:bench -- --quick              # local before/after numbers
    bazel run -c opt //tools/bench:bench -- --out DIR     # the nightly job

Without ``--out`` the files go to ``bench-results/`` in the workspace (under
``bazel run``) or the current directory. Numbers are printed as a table too.
Nothing here judges a number: the nightly history is advisory (Q64, Q71).
"""

from __future__ import annotations

import argparse
import os
import pkgutil
import sys
import time
from collections.abc import Callable
from importlib import import_module
from pathlib import Path

import ipu_bench.benchmarks
from ipu_bench.result import Budget, Result, write_results


def benchmark_modules() -> dict[str, Callable[[bool], list[Result]]]:
    """Every module in :mod:`ipu_bench.benchmarks`, by short name."""
    modules = {}
    for info in sorted(pkgutil.iter_modules(ipu_bench.benchmarks.__path__), key=lambda m: m.name):
        if info.name.startswith("_"):
            continue
        module = import_module(f"ipu_bench.benchmarks.{info.name}")
        modules[info.name] = module.run
    return modules


def run_all(budget: Budget, only: list[str] | None = None) -> list[Result]:
    modules = benchmark_modules()
    unknown = sorted(set(only or ()) - modules.keys())
    if unknown:
        raise SystemExit(f"unknown benchmark module(s) {unknown}; known: {sorted(modules)}")
    results = []
    for name, run in modules.items():
        if only and name not in only:
            continue
        start = time.perf_counter()
        found = run(budget)
        if not found or not all(isinstance(r, Result) for r in found):
            raise RuntimeError(f"benchmark module {name} returned no Result list: {found!r}")
        print(f"[{name}] {len(found)} results in {time.perf_counter() - start:.1f} s", flush=True)
        results += found
    return results


def format_table(results: list[Result]) -> str:
    width = max(len(r.name) for r in results)
    lines = []
    for r in results:
        direction = "higher is better" if r.bigger_is_better else "lower is better"
        lines.append(f"{r.name:<{width}}  {r.value:>14.4f} {r.unit:<9} ({direction})  {r.extra}")
    return "\n".join(lines)


def _default_out() -> Path:
    workspace = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    return Path(workspace) / "bench-results" if workspace else Path.cwd()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--quick", action="store_true", help="fewer repeats, for local use")
    size.add_argument(
        "--tiny", action="store_true", help="as little as still runs each module (smoke test)"
    )
    parser.add_argument("--out", type=Path, help="directory for the two JSON files")
    parser.add_argument(
        "--only", action="append", metavar="MODULE", help="run only this module (repeatable)"
    )
    args = parser.parse_args(argv)
    budget = Budget.TINY if args.tiny else Budget.QUICK if args.quick else Budget.FULL
    results = run_all(budget, args.only)
    smaller, bigger = write_results(results, args.out or _default_out())
    print(format_table(results))
    print(f"wrote {smaller}\nwrote {bigger}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
