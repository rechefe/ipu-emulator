"""No assembly source tests definedness or falls back with `| default` (E05-11).

Templates render with StrictUndefined: a value a template needs is a declared
`ipu:params` parameter, and its default lives in the header. `| default(...)`,
`is defined` and `is undefined` would bring back the silent fallback.

The check is textual, so a comment that says "is defined" in prose fails it
too; reword the comment. It reads every program source in the repo: the kernel
tree (``kernels/``), the ``ipu_apps`` kernels and MobileViT programs until
E07-25 retires them, and the editor's corpus.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path.cwd()

#: Repo-relative directories whose program sources (recursively) are linted.
#: Under Bazel only the files in the test's data are there: the editor's
#: negative corpus (texts that do not assemble) is left out.
SOURCE_ROOTS = (
    "kernels",
    "src/tools/ipu-apps/src/ipu_apps/kernels",
    "src/tools/ipu-apps/test/mobilevit",
    "vscode-ipu-asm/test",
)

#: A Jinja construct that only makes sense when a name may be undefined.
_FALLBACK = re.compile(r"\|\s*(?:default|d)\b|\bis\s+(?:not\s+)?(?:un)?defined\b")


def fallbacks(text: str) -> list[int]:
    """The 1-based lines of `text` that use a definedness fallback."""
    return [n for n, line in enumerate(text.splitlines(), start=1) if _FALLBACK.search(line)]


def _sources() -> list[Path]:
    return sorted(
        path
        for root in SOURCE_ROOTS
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
        "{{ rows | d(8) }}",
        "{{ rows | default }}",
        "{% if rows is defined %}",
        "{% if rows is not defined %}",
        "{% if rows is undefined %}",
    ],
)
def test_a_planted_fallback_is_found(line):
    assert fallbacks(f"HALT;;\n{line}\n") == [2]


def test_the_corpus_is_scanned():
    assert _sources(), "the corpus glob found no programs"
