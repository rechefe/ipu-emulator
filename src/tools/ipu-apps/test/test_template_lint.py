"""No assembly source tests definedness or falls back with `| default` (E05-11).

Templates render with StrictUndefined: a value a template needs is a declared
`ipu:params` parameter, and its default lives in the header. `| default(...)`,
`is defined` and `is undefined` would bring back the silent fallback.
"""

from __future__ import annotations

import re
from pathlib import Path

import asm_corpus
import pytest

ROOT = Path.cwd()

#: A Jinja construct that only makes sense when a name may be undefined.
_FALLBACK = re.compile(r"\|\s*default\s*\(|\bis\s+(?:not\s+)?(?:un)?defined\b")


def fallbacks(text: str) -> list[int]:
    """The 1-based lines of `text` that use a definedness fallback."""
    return [n for n, line in enumerate(text.splitlines(), start=1) if _FALLBACK.search(line)]


def _sources() -> list[Path]:
    return sorted(
        path
        for root in asm_corpus.CORPUS_ROOTS
        for pattern in ("*.asm", "*.asm.j2")
        for path in (ROOT / root).rglob(pattern)
    )


@pytest.mark.parametrize("path", _sources(), ids=lambda p: p.relative_to(ROOT).as_posix())
def test_no_source_falls_back_on_an_undefined_name(path):
    lines = fallbacks(path.read_text(encoding="utf-8"))
    assert not lines, f"{path.relative_to(ROOT)}: lines {lines}: declare an ipu:params default"


@pytest.mark.parametrize(
    "line",
    [
        "{%- set n = rows | default(8) -%}",
        "{%- set n = rows|default (8) -%}",
        "{% if rows is defined %}",
        "{% if rows is not defined %}",
        "{% if rows is undefined %}",
    ],
)
def test_a_planted_fallback_is_found(line):
    assert fallbacks(f"BKPT;;\n{line}\n") == [2]


def test_the_corpus_is_scanned():
    assert len(_sources()) > 50
