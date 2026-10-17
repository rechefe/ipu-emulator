"""Every fenced `asm` block in the docs assembles: the hand-written pages and
the generated ISA reference (E09-14), where a failing example is a generator bug.

A block that is deliberately incomplete is marked with
`<!-- asm-fragment: reason -->` on the line before its fence; it is reported as
skipped with that reason. A kernel's program, marked
`<!-- asm-kernel: kernels/<family>/<name> -->`, is assembled as that kernel's
program: rendered with its default case and its `[abi]` names, as the harness
renders it.
"""

from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path

import pytest

from docs.doc_blocks import Block, DocBlockError, fenced_blocks
from docs.doc_files import doc_files, generated_files
from ipu_as import diagnostics
from ipu_harness.assembly import assemble
from ipu_harness.cases import expand
from ipu_harness.manifest import load_manifest
from ipu_harness.params import plan_case
from ipu_isa.errors import IpuError


def _cases() -> list:
    # The generated pages come from the test's data; without them it would check nothing.
    assert generated_files(), "no generated ISA reference pages in the test data"
    cases = []
    for path in [*doc_files(), *generated_files()]:
        try:
            blocks = fenced_blocks(path, "asm")
        except DocBlockError as error:
            cases.append(pytest.param(error, id=str(path)))
            continue
        cases.extend(pytest.param(block, id=f"{block.path}:{block.line}") for block in blocks)
    return cases


def _assemble_as_kernel(case: Block) -> None:
    """The block as its kernel's program, through the harness (E06-07)."""
    manifest = load_manifest(Path(case.kernel or ""))
    with tempfile.TemporaryDirectory() as scratch:
        program = Path(scratch) / manifest.asm.name
        program.write_text(case.body, encoding="utf-8")
        manifest = dataclasses.replace(manifest, asm=program)
        default = next(c for c in expand(manifest) if c.decl is manifest.cases["default"])
        for fmt in default.formats:
            try:
                assemble(plan_case(manifest, default.values, fmt, case=default.decl))
            except IpuError as error:
                pytest.fail(f"{case.path}:{case.line}: {fmt.value}: {error}", pytrace=False)


@pytest.mark.parametrize("case", _cases())
def test_asm_block_assembles(case: Block | DocBlockError) -> None:
    if isinstance(case, DocBlockError):
        pytest.fail(str(case))
    if case.fragment_reason is not None:
        pytest.skip(f"asm-fragment: {case.fragment_reason}")
    if case.kernel is not None:
        _assemble_as_kernel(case)
        return
    errors = [d for d in diagnostics.check(case.body) if d.severity == "error"]
    if errors:
        first = errors[0]
        # Diagnostic lines are 0-based within the body; the body starts on the
        # line after the fence.
        doc_line = case.line + 1 + first.line
        pytest.fail(f"{case.path}:{doc_line}: {first.message}", pytrace=False)
