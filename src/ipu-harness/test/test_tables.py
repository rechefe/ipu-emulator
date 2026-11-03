"""XMEM table declarations (``[tables]``) held to the runs of the scale kernel."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
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


# -- multi-stage kernels: [tables.stage<N>] ---------------------------------------------------

#: Stage 1 copies x into the scratch t, stage 2 scales t into y (a handoff buffer).
COPY = """\
    SET LR0 CR0 ;
    SET LR1 CR0 ;;
row_loop:
    LD.MULT R0 LR1 {{ cr.x }} ;;
    MULT.VE LR0 CR1 0 LR0 CR15.L {{ cr.qf }}.L ;
    ACC.ADD.FIRST ;;
    ACTIVATE.QUANTIZE identity CR15.L {{ cr.qf }}.L {{ cr.t }} LR1 ;;
    ADD LR1 LR1 CR1 ;;
    BLT.S LR1 {{ cr.rows }} row_loop ;;
    HALT ;;
"""
SCALE_T = ASM.replace("{{ cr.x }}", "{{ cr.t }}")
STAGES_HOOK = """\
def run(ctx):
    ctx.run_stage("copy.asm", abi={"x": "CR2", "t": "CR3", "rows": "CR4", "qf": "CR14"})
    ctx.run_stage(
        "scale_t.asm", abi={"t": "CR2", "y": "CR3", "rows": "CR4", "s": "CR5", "qf": "CR6"}
    )
"""
STAGES_TOML = """\
[kernel]
op = "copy_scale"
formats = ["e4"]
asm = "copy.asm"
stages = "stages.py:run"

[params]
rows = { type = "int", min = 1, max = 8 }
s    = { type = "float" }

[inputs.x]
shape = ["rows", 128]

[scratch.t]
shape = ["rows", 128]

[outputs.y]
shape = ["rows", 128]

[derived]
qf = "case_qf"

[abi]
x    = "CR2"
t    = "CR3"
y    = "CR6"
rows = "CR4"
s    = "CR5"
qf   = "CR14"

[golden]
tolerance = { e4 = "EXACT" }

[cases.default]
rows = 3
s = 2.0
inputs = { x = "data/x.npy" }
expected = { y = "data/y.npy" }
"""
STAGE_TABLES = """
[tables.stage1]
x = {}
t = {}

[tables.stage2]
t = {}
y = {}
"""
E4 = variant("e4")


def _staged(tmp_path: Path, tables: str, *, scale_asm: str = SCALE_T) -> KernelManifest:
    directory = tmp_path / "elementwise" / "copy_scale"
    (directory / "data").mkdir(parents=True)
    x = np.arange(3 * 128, dtype=np.float64).reshape(3, 128) % 17 - 8
    np.save(directory / "data" / "x.npy", x)
    np.save(directory / "data" / "y.npy", x * 2.0)
    (directory / "copy.asm").write_text(COPY)
    (directory / "scale_t.asm").write_text(scale_asm)
    (directory / "stages.py").write_text(STAGES_HOOK)
    (directory / "kernel.toml").write_text(STAGES_TOML + tables)
    return load_manifest(directory)


def _run_e4(manifest: KernelManifest) -> RunResult:
    case = next(c for c in expand(manifest) if c.name == "default")
    return run_kernel(manifest, case, E4, RunOptions(tables=True))


def test_each_stage_is_held_to_its_own_tables(tmp_path):
    result = _run_e4(_staged(tmp_path, STAGE_TABLES))
    assert result.passed
    assert result.tables is not None
    assert result.tables.findings == ()
    # rows = 3: t is written by stage 1 and read by stage 2, each its own profile.
    assert result.tables.profiles["stage1.t"].writes == 3
    assert result.tables.profiles["stage2.t"].reads == 3
    assert sorted(result.tables.profiles) == ["stage1.t", "stage1.x", "stage2.t", "stage2.y"]


def test_a_region_a_stage_leaves_undeclared_is_an_access_through_no_table(tmp_path):
    result = _run_e4(
        _staged(tmp_path, "[tables.stage1]\nx = {}\nt = {}\n[tables.stage2]\nt = {}\n")
    )
    assert result.tables is not None
    assert [f.describe() for f in result.tables.errors] == [
        "error: CR3: stage2: 3 accesses through CR3, which no declared table is reached through "
        "(rows 3..5, in ['y'])"
    ]


def test_a_handoff_buffer_has_each_stages_own_numbers(tmp_path):
    prefetch = SCALE_T.replace("    HALT ;;", "    LD.MULT R0 LR1 {{ cr.t }} ;;\n    HALT ;;")
    tables = STAGE_TABLES.replace(
        "[tables.stage2]\nt = {}", "[tables.stage2]\nt = { overhead_rows = 1 }"
    )
    result = _run_e4(_staged(tmp_path, tables, scale_asm=prefetch))
    assert result.tables is not None and result.tables.errors == ()
    assert result.placement is not None
    # Only stage 2 reaches row 3 of t; the region reserves it, so y starts after it.
    assert result.placement.region("t").overhead_rows == 1


def test_an_access_past_a_stages_rows_is_an_error_even_if_another_stage_declares_overhead(tmp_path):
    prefetch = COPY.replace("    HALT ;;", "    LD.MULT R0 LR1 {{ cr.t }} ;;\n    HALT ;;")
    tables = STAGE_TABLES.replace(
        "[tables.stage2]\nt = {}", "[tables.stage2]\nt = { overhead_rows = 1 }"
    )
    directory = _staged(tmp_path, tables).directory
    (directory / "copy.asm").write_text(prefetch)
    result = _run_e4(load_manifest(directory))
    assert result.tables is not None
    (message,) = [f.describe() for f in result.tables.errors]
    assert message.startswith("error: stage1.t: 1 accesses outside rows=3 + overhead_rows=0")


def test_a_declared_stage_that_does_not_run_is_an_error(tmp_path):
    result = _run_e4(_staged(tmp_path, STAGE_TABLES + "\n[tables.stage3]\nt = {}\n"))
    assert result.tables is not None
    assert [f.describe() for f in result.tables.errors] == [
        "error: stage3: declared in [tables], but the stages hook ran no such stage"
    ]


def test_a_table_of_a_tensor_the_stage_does_not_bind_is_an_error(tmp_path):
    result = _run_e4(_staged(tmp_path, STAGE_TABLES + "y = {}\n".replace("y", "x")))
    assert result.tables is not None
    assert "error: stage2.x: reached through no CR in this stage" in [
        f.describe() for f in result.tables.errors
    ]


def test_the_suggestion_for_a_staged_kernel_has_a_section_per_stage(tmp_path):
    result = _run_e4(_staged(tmp_path, ""))
    assert result.tables is not None
    assert result.tables.suggestion() == (
        "[tables.stage1]\nx = {}\nt = {}\n[tables.stage2]\ny = {}\nt = {}"
    )


@pytest.mark.parametrize(
    ("tables", "error"),
    [
        ("[tables]\nx = {}\n", "[tables.stage<N>] sections, not 'x'"),
        ("[tables.stage0]\nx = {}\n", "not 'stage0'"),
        ("[tables.stage1]\nz = {}\n", "a table is a region; 'z' is no tensor"),
        ("[tables.stage1]\nx = { also_via = [] , rows = 1 }\n", "rows"),
        ('[tables.stage1]\nx = { also_via = ["CR0"] }\n', "takes no also_via"),
    ],
)
def test_a_wrong_stage_declaration_does_not_load(tmp_path, tables, error):
    with pytest.raises(ManifestError, match=re.escape(error)):
        _staged(tmp_path, tables)


def test_a_table_reached_through_a_derived_base_is_pending(tmp_path):
    manifest = _kernel(tmp_path, "[tables]\nx = {}\ny = {}\n")
    toml = manifest.directory / "kernel.toml"
    text = toml.read_text().replace('qf = "case_qf"', 'qf = "case_qf"\nxb = "base(x)"')
    toml.write_text(text.replace('x    = "CR2"', 'xb   = "CR2"'))
    asm = manifest.directory / "scale.asm"
    asm.write_text(asm.read_text().replace("cr.x", "cr.xb"))
    result = _run(load_manifest(manifest.directory))
    assert result.passed
    assert _messages(result) == []
    assert _messages(result, "pending") == [
        "pending: x: CR2 holds xb, a value derived from its base"
    ]


def test_a_pending_table_turns_its_errors_into_pending_findings(tmp_path):
    tables = '[tables]\nx = { pending = "prefetches a row past it" }\ny = {}\n'
    result = _run(_kernel(tmp_path, tables, asm=PREFETCH_ASM))
    assert result.passed
    assert _messages(result) == []
    (message,) = _messages(result, "pending")
    assert message.startswith("pending: x: prefetches a row past it: 1 accesses outside rows=4")


def test_a_pending_table_with_nothing_wrong_is_an_error(tmp_path):
    result = _run(_kernel(tmp_path, '[tables]\nx = { pending = "old reason" }\ny = {}\n'))
    assert _messages(result) == ["error: x: pending, but nothing is wrong any more: delete the key"]
