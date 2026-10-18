"""``assemble/{render,parse,encode}_s``: the assembler pipeline split by stage.

Over every kernel's default program (rendered by the harness, so the render
stage is the Jinja pass over text with no template left) and every program in
the E02-01 digest manifest, as ``bazel run //src/tools/ipu-as-py:bench_assemble``
measures it (E05-01): the parser built and the tables warm first, best of 3
runs (1 with ``--quick``).
"""

from __future__ import annotations

from ipu_bench.isolated import call
from ipu_bench.result import Budget, Result


def measure(repeat: int, limit: int | None = None) -> dict:
    """In a fresh interpreter: the stage totals ``bench_assemble.measure`` returns."""
    import tempfile
    from pathlib import Path

    import bench_assemble

    from ipu_bench.kernel_runs import default_programs

    root = bench_assemble.find_root()
    programs = bench_assemble.load_programs(root)
    with tempfile.TemporaryDirectory(prefix="ipu-bench-") as tmp:
        for name, text, words in default_programs():
            path = Path(tmp, name + ".asm")
            path.write_text(text)
            programs.append(bench_assemble.Program(name, str(path), None, words))
        # An absolute path joins the root as itself.
        return bench_assemble.measure(root, programs[:limit], repeat=repeat)


def run(budget: Budget) -> list[Result]:
    m = call(
        "ipu_bench.benchmarks.assemble_stages:measure",
        repeat=3 if budget is Budget.FULL else 1,
        limit=3 if budget is Budget.TINY else None,
    )
    extra = f"files={m['files']} words={m['words']}"
    return [
        Result(f"assemble/{stage}_s", "s", m[stage], extra=extra)
        for stage in ("render", "parse", "encode")
    ]
