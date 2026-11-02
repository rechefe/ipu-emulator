"""XMEM table declarations (``[tables]``) held to the runs of the scale kernel."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from scale_kernel import ASM, write_scale

from ipu_harness.cases import expand
from ipu_harness.manifest import KernelManifest, ManifestError, load_manifest
from ipu_harness.runner import RunOptions, RunResult, run_kernel
from ipu_harness.variants import variant

FP32 = variant("fp32")

#: scale.asm with one more load after the loop: row ``rows`` of x, one past it
#: (as a loop's last prefetch reads).
PREFETCH_ASM = ASM.replace("    HALT ;;", "    LD.MULT R0 LR1 {{ cr.x }} ;;\n    HALT ;;")


def _kernel(tmp_path: Path, tables: str, *, asm: str = ASM, extra: str = "") -> KernelManifest:
    """The scale kernel (rows = 4 by default) with ``tables`` appended to its kernel.toml."""
    manifest = write_scale(tmp_path, with_reference=False, asm=asm)
    toml = manifest.directory / "kernel.toml"
    text = toml.read_text().replace(
        'formats = ["fp32", "e4"]', f'formats = ["fp32", "e4"]\n{extra}'
    )
    toml.write_text(text + "\n" + tables)
    return load_manifest(manifest.directory)


def _run(manifest: KernelManifest, case: str = "default") -> RunResult:
    chosen = next(c for c in expand(manifest) if c.name == case)
    return run_kernel(manifest, chosen, FP32, RunOptions(tables=True))


def _messages(result: RunResult, severity: str = "error") -> list[str]:
    assert result.tables is not None
    return [f.describe() for f in result.tables.findings if f.severity == severity]


def test_a_matching_declaration_has_no_finding(tmp_path):
    result = _run(_kernel(tmp_path, "[tables]\nx = {}\ny = {}\n"))
    assert result.passed
    assert result.tables is not None
    assert result.tables.findings == ()
    x = result.tables.profiles["x"]
    # rows = 4: x is read once, rows 0..3 in order; y is written likewise.
    assert (x.reads, x.writes, x.lowest, x.highest, x.repetitions) == (4, 0, 0, 3, 1)
    assert result.tables.profiles["y"].writes == 4


def test_a_region_left_undeclared_is_an_access_through_no_table(tmp_path):
    result = _run(_kernel(tmp_path, "[tables]\nx = {}\n"))
    assert _messages(result) == [
        "error: CR3: 4 accesses through CR3, which no declared table is reached through "
        "(rows 4..7, in ['y'])"
    ]


def test_other_repetitions_than_declared_fail(tmp_path):
    result = _run(_kernel(tmp_path, '[tables]\nx = { repetitions = "rows - 2" }\ny = {}\n'))
    (message,) = _messages(result)
    assert message.startswith("error: x: 1 repetitions, declared 2")


def test_an_access_past_the_rows_needs_overhead_rows(tmp_path):
    result = _run(_kernel(tmp_path, "[tables]\nx = {}\ny = {}\n", asm=PREFETCH_ASM))
    (message,) = _messages(result)
    assert message.startswith(
        "error: x: 1 accesses outside rows=4 + overhead_rows=0; "
        "covering them takes overhead_rows=1; row 4 at cycle"
    )


def test_overhead_rows_are_reserved_and_checked_as_the_tables(tmp_path):
    result = _run(
        _kernel(tmp_path, "[tables]\nx = { overhead_rows = 1 }\ny = {}\n", asm=PREFETCH_ASM)
    )
    assert result.passed
    assert result.tables is not None and result.tables.errors == ()
    assert result.placement is not None
    # x holds rows 0..3 and reserves row 4, so y starts at row 5, not 4.
    assert result.placement.region("x").overhead_rows == 1
    assert result.placement.region("y").base_row == 5


def test_declared_overhead_the_run_never_reaches_is_a_warning(tmp_path):
    result = _run(_kernel(tmp_path, "[tables]\nx = { overhead_rows = 2 }\ny = {}\n"))
    assert _messages(result) == []
    assert _messages(result, "warning") == ["warning: x: overhead_rows=2; the run reaches 0"]


def test_a_kernel_without_tables_gets_a_suggestion(tmp_path):
    manifest = write_scale(tmp_path, with_reference=False, asm=PREFETCH_ASM)
    result = _run(manifest)
    assert result.tables is not None
    assert result.tables.findings == ()
    assert result.tables.suggestion() == "[tables]\nx = { overhead_rows = 1 }\ny = {}"


def test_tables_from_a_python_hook(tmp_path):
    hook = 'def tables(plan):\n    return {"x": {"overhead_rows": 1}, "y": {}}\n'
    manifest = write_scale(tmp_path, with_reference=False, asm=PREFETCH_ASM)
    (manifest.directory / "tables.py").write_text(hook)
    toml = manifest.directory / "kernel.toml"
    toml.write_text(
        toml.read_text().replace('op = "scale"', 'op = "scale"\ntables = "tables.py:tables"')
    )
    result = _run(load_manifest(manifest.directory))
    assert result.tables is not None and result.tables.findings == ()
    assert result.placement is not None and result.placement.region("y").base_row == 5


@pytest.mark.parametrize(
    ("tables", "error"),
    [
        ("[tables]\nz = {}\n", "a table is a region; 'z' is no tensor"),
        ("[tables]\nrows = {}\n", "a table is a region; 'rows' is no tensor"),
        ("[tables]\nx = { whole_run = true, repetitions = 2 }\n", "takes no repetitions"),
        ('[tables]\nx = { also_via = ["s"] }\n', "does not read base(x)"),
        ('[tables]\nx = { jump_back = "nope" }\n', "names no parameter"),
        ("[tables]\nx = { rows = 4 }\n", "rows"),
    ],
)
def test_a_wrong_declaration_does_not_load(tmp_path, tables, error):
    with pytest.raises(ManifestError, match=re.escape(error)):
        _kernel(tmp_path, tables)
