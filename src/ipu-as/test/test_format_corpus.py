"""Formatting every kernel is safe (E08-24): idempotent, and the same program.

Report mode: a kernel that formatting would change is listed (a warning in
the test log), not failed, since this is no gate on kernel style. The test fails only
when formatting would change a program, or formatting twice differs from once.

A kernel without Jinja must assemble to the same words before and after; a
template, which needs its defines to render, must read as the same stream of
Jinja tags, comments and tokens (`ipu_as.formatter.check_same_program`, which
`format_source` also runs).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from ipu_as import template
from ipu_as.errors import FormatError
from ipu_as.formatter import check_same_program, format_source
from ipu_as.session import AssemblySession

# The workspace root: the runfiles tree under Bazel, the checkout under plain pytest.
ROOT = Path(__file__).parents[3]
KERNELS = sorted((ROOT / "kernels").glob("**/*.asm"))
OPTIONS = {"default": {}, "blank-line": {"blank_line": True}}


def _words(text: str) -> tuple[int, ...]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return AssemblySession(text).assemble().words


def test_the_corpus_is_there():
    assert len(KERNELS) > 50


@pytest.mark.parametrize("path", KERNELS, ids=lambda p: str(p.relative_to(ROOT)))
@pytest.mark.parametrize("variant", sorted(OPTIONS))
def test_formatting_a_kernel_keeps_its_program(path, variant):
    text = path.read_text(encoding="utf-8")
    try:
        once = format_source(text, **OPTIONS[variant])
    except FormatError as error:
        pytest.fail(f"{path.relative_to(ROOT)}: {error}")
    assert format_source(once, **OPTIONS[variant]) == once, "formatting twice differs from once"
    check_same_program(text, once)
    if not template.has_markers(text):
        assert _words(once) == _words(text)


def test_report_the_kernels_formatting_would_change():
    """Never fails: lists, as a warning in the test log, each kernel not in the layout."""
    changed = [
        str(path.relative_to(ROOT))
        for path in KERNELS
        if format_source(text := path.read_text(encoding="utf-8")) != text
    ]
    if changed:
        listed = "".join(f"\n  {name}" for name in changed)
        warnings.warn(f"ipu-as format would change {len(changed)} kernel(s):{listed}", stacklevel=1)
