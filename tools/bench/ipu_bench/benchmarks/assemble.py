"""``assemble/*``: assemble every kernel's default program once through the library API.

The programs are the kernel tree's, rendered for each kernel's default case
(the program ``:assemble_<name>`` writes); only ``lark_tree.assemble`` is timed.
"""

from __future__ import annotations

import time

from ipu_bench.isolated import call
from ipu_bench.result import Budget, Result


def measure(limit: int | None = None) -> dict:
    """In a fresh interpreter: parser construction included, rendering excluded."""
    import warnings

    from ipu_as.lark_tree import assemble
    from ipu_bench.kernel_runs import default_programs

    texts = [text for _, text, _ in default_programs()[:limit]]
    if not texts:
        raise RuntimeError("no kernel programs found (is the kernel tree in the runfiles?)")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*-only .* slot")
        start = time.perf_counter()
        words = sum(len(assemble(text)) for text in texts)
        seconds = time.perf_counter() - start
    return {"files": len(texts), "words": words, "seconds": seconds}


def run(budget: Budget) -> list[Result]:
    m = call("ipu_bench.benchmarks.assemble:measure", limit=3 if budget is Budget.TINY else None)
    extra = f"files={m['files']} words={m['words']}"
    return [
        Result("assemble/all_kernels", "s", m["seconds"], extra=extra),
        Result(
            "assemble/words_per_s",
            "words/s",
            m["words"] / m["seconds"],
            extra=extra,
            bigger_is_better=True,
        ),
    ]
