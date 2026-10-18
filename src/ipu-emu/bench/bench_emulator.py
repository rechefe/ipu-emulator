"""Emulator benchmarks for the Q71 targets (E04-02): only the ``emulator`` module of //tools/bench.

    bazel run //src/ipu-emu:bench_emulator -- --quick                 # one run each
    bazel run --config=proto //src/ipu-emu:bench_emulator -- --quick  # adds prototyping mode

Writes ``bench-smaller.json`` / ``bench-bigger.json`` like ``//tools/bench:bench``
(``--out DIR``; default ``bench-results/`` in the workspace) and prints a table.
What is measured: ``tools/bench/README.md``.
"""

from __future__ import annotations

import sys

from ipu_bench import bench

if __name__ == "__main__":
    sys.exit(bench.main(["--only", "emulator", *sys.argv[1:]]))
