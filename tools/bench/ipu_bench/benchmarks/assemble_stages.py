"""``assemble/{render,parse,encode}_s``: the assembler pipeline split by stage.

Over every program in the E02-01 digest manifest, as
``bazel run //src/tools/ipu-as-py:bench_assemble`` measures it (E05-01): the
parser built and the tables warm first, best of 3 runs (1 with ``--quick``).
"""

from __future__ import annotations

from ipu_bench.isolated import call
from ipu_bench.result import Result, tiny_budget


def measure(repeat: int, limit: int | None = None) -> dict:
    """In a fresh interpreter: the stage totals ``bench_assemble.measure`` returns."""
    import bench_assemble

    root = bench_assemble.find_root()
    programs = bench_assemble.load_programs(root)[:limit]
    return bench_assemble.measure(root, programs, repeat=repeat)


def run(quick: bool) -> list[Result]:
    tiny = tiny_budget()
    m = call(
        "ipu_bench.benchmarks.assemble_stages:measure",
        repeat=1 if quick or tiny else 3,
        limit=3 if tiny else None,
    )
    extra = f"files={m['files']} words={m['words']}"
    return [
        Result(f"assemble/{stage}_s", "s", m[stage], extra=extra)
        for stage in ("render", "parse", "encode")
    ]
