"""``assemble/*``: assemble every kernel ``.asm`` once through the library API."""

from __future__ import annotations

import time
from pathlib import Path

from ipu_bench.isolated import call
from ipu_bench.result import Budget, Result


def kernel_sources() -> list[Path]:
    import ipu_apps.kernels

    root = Path(next(iter(ipu_apps.kernels.__path__)))
    return sorted(root.rglob("*.asm"))


def measure(limit: int | None = None) -> dict:
    """In a fresh interpreter: parser construction included, file reading excluded."""
    from ipu_as.lark_tree import assemble

    texts = [path.read_text() for path in kernel_sources()[:limit]]
    if not texts:
        raise RuntimeError("no kernel .asm files found (are they in the runfiles?)")
    start = time.perf_counter()
    words = sum(len(assemble(text)) for text in texts)
    return {"files": len(texts), "words": words, "seconds": time.perf_counter() - start}


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
