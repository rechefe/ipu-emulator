"""Assembler wall time per pipeline stage over the assembled-binary corpus.

    bazel run //src/tools/ipu-as-py:bench_assemble               # best of 3
    bazel run //src/tools/ipu-as-py:bench_assemble -- --repeat 5 --json

Assembles every program the E02-01 digest manifest
(``src/tools/ipu-apps/test/baselines/asm_digests.json``) lists, and splits the
wall time into the three stages of ``ipu_as.session.AssemblySession``:

- render: the Jinja layer (for a manifest entry with a render context, the
  ``jinja2.Template`` render the digest test does, plus the session's render);
- parse: grammar and AST building, labels included;
- encode: building and encoding every compound instruction word.

The parser is built and the ISA tables are warmed by one untimed assembly
first, so the numbers are per-run costs. Each stage total is the best of
``--repeat`` runs. A program whose word count differs from the manifest is an
error: the numbers would not be for the pinned corpus.

The numbers are tracked, not gated (Q64, Q71). The nightly benchmark job
records them as ``assemble/{render,parse,encode}_s`` through
``//tools/bench`` (``ipu_bench.benchmarks.assemble_stages``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import jinja2

from ipu_as.session import AssemblySession

#: The E02-01 digest manifest, repo-relative.
MANIFEST_PATH = "src/tools/ipu-apps/test/baselines/asm_digests.json"

STAGES = ("render", "parse", "encode")


@dataclass(frozen=True)
class Program:
    """One manifest entry: a repo-relative `.asm` path and its render context."""

    key: str
    path: str
    context: dict | None
    words: int


def find_root(start: Path | None = None) -> Path:
    """The directory the manifest's paths are relative to: the runfiles root
    under ``bazel run``/``bazel test``, else the checkout holding this file."""
    candidates = [start or Path.cwd(), *Path(__file__).resolve().parents]
    for candidate in candidates:
        if (candidate / MANIFEST_PATH).is_file():
            return candidate
    raise FileNotFoundError(f"{MANIFEST_PATH} not found from {candidates[0]}")


def load_programs(root: Path) -> list[Program]:
    """Every manifest entry, in manifest (sorted key) order."""
    manifest = json.loads((root / MANIFEST_PATH).read_text())
    programs = []
    for key, entry in sorted(manifest.items()):
        path, _, context_json = key.partition("?")
        context = json.loads(context_json) if context_json else None
        programs.append(Program(key, path, context, entry["words"]))
    return programs


def _assemble_timed(text: str, program: Program) -> tuple[float, float, float, int]:
    """Seconds spent rendering, parsing and encoding one program, and its word count."""
    t0 = time.perf_counter()
    if program.context is not None:
        text = jinja2.Template(text).render(**program.context)
    session = AssemblySession(text, name=program.path)
    session.render()
    t1 = time.perf_counter()
    session.parse()
    t2 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        words = len(session.assemble().words)
    t3 = time.perf_counter()
    return t1 - t0, t2 - t1, t3 - t2, words


def measure(root: Path, programs: list[Program], repeat: int = 3) -> dict:
    """Best-of-``repeat`` seconds per stage over all ``programs``.

    Returns ``{"files", "words", "render", "parse", "encode", "total"}``.
    """
    if not programs:
        raise ValueError("no programs to assemble")
    texts = [(root / program.path).read_text() for program in programs]
    _assemble_timed(texts[0], programs[0])  # build the parser, warm the tables
    best = dict.fromkeys(STAGES, float("inf"))
    words = 0
    for _ in range(max(1, repeat)):
        totals = dict.fromkeys(STAGES, 0.0)
        words = 0
        for text, program in zip(texts, programs, strict=True):
            *seconds, count = _assemble_timed(text, program)
            if count != program.words:
                raise RuntimeError(
                    f"{program.key}: {count} words, the manifest records {program.words}"
                )
            words += count
            for stage, value in zip(STAGES, seconds, strict=True):
                totals[stage] += value
        for stage in STAGES:
            best[stage] = min(best[stage], totals[stage])
    return {
        "files": len(programs),
        "words": words,
        **best,
        "total": sum(best[stage] for stage in STAGES),
    }


def format_table(result: dict) -> str:
    lines = [
        f"{result['files']} programs, {result['words']} words",
        f"{'stage':<8}{'seconds':>10}{'share':>8}",
    ]
    for stage in (*STAGES, "total"):
        share = result[stage] / result["total"] if result["total"] else 0.0
        lines.append(f"{stage:<8}{result[stage]:>10.3f}{share:>8.0%}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repeat", type=int, default=3, help="best of N runs (default 3)")
    parser.add_argument("--limit", type=int, default=None, help="only the first N programs")
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    args = parser.parse_args(argv)
    root = find_root()
    programs = load_programs(root)[: args.limit]
    result = measure(root, programs, repeat=args.repeat)
    print(json.dumps(result, indent=2) if args.json else format_table(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
