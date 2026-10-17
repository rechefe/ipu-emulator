"""Every fenced `asm` block in the docs assembles: the hand-written pages and
the generated ISA reference (E09-14), where a failing example is a generator bug.

A block that is deliberately incomplete is marked with
`<!-- asm-fragment: reason -->` on the line before its fence; it is reported as
skipped with that reason.
"""

from __future__ import annotations

import pytest

from docs.doc_blocks import Block, DocBlockError, fenced_blocks
from docs.doc_files import doc_files, generated_files
from ipu_as import diagnostics


def _cases() -> list:
    cases = []
    for path in [*doc_files(), *generated_files()]:
        try:
            blocks = fenced_blocks(path, "asm")
        except DocBlockError as error:
            cases.append(pytest.param(error, id=str(path)))
            continue
        cases.extend(pytest.param(block, id=f"{block.path}:{block.line}") for block in blocks)
    return cases


@pytest.mark.parametrize("case", _cases())
def test_asm_block_assembles(case: Block | DocBlockError) -> None:
    if isinstance(case, DocBlockError):
        pytest.fail(str(case))
    if case.fragment_reason is not None:
        pytest.skip(f"asm-fragment: {case.fragment_reason}")
    errors = [d for d in diagnostics.check(case.body) if d.severity == "error"]
    if errors:
        first = errors[0]
        # Diagnostic lines are 0-based within the body; the body starts on the
        # line after the fence.
        doc_line = case.line + 1 + first.line
        pytest.fail(f"{case.path}:{doc_line}: {first.message}", pytrace=False)
