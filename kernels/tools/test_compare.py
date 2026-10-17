"""compare_setup and compare_legacy on reshape/identity, rebuilt as a kernel.toml kernel (E07-01).

The new kernel is what a migration writes: the old ``identity.asm`` unchanged,
a ``kernel.toml`` whose ``[abi]`` and placement reproduce the old CR map, and a
generator that keeps the old case's input values. Each test plants one change
in a copy and checks that the tool names it.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import compare_legacy
import compare_setup
import pytest
from compare_legacy import compare_family, table
from compare_setup import compare_setup as setup_report
from migration import legacy_asm_path, runfile

from ipu_apps.kernel_registry.registry import kernel_spec

TOML = """\
[kernel]
op = "identity"
formats = ["fp32"]

[params]
rows = { type = "int", min = 1 }

[inputs.x]
shape = ["rows", 128]

[outputs.y]
shape = ["rows", 128]

[abi]
x    = "CR2"
y    = "CR3"
rows = "CR4"

[cases.default]
rows = 3
generator = "gen.py:make"

[cases.single_row]
rows = 1
generator = "gen.py:make"
"""

# The old cases.py's values: arange(rows * 128) - 128, in FP32.
GENERATOR = """\
import numpy as np


def make(plan, rng):
    rows = plan.params["rows"]
    values = np.arange(rows * 128, dtype=np.float32) - np.float32(128)
    {plant}
    return {{"x": values.reshape(rows, 128)}}
"""

#: The one immediate the binary plant changes: the MULT.VE mask offset, 0 -> 1.
IMMEDIATE = "MULT.VE {{lr_zero}} cr1 0 "


@pytest.fixture
def tree(tmp_path) -> Path:
    """A kernel tree holding the new identity kernel."""
    folder = tmp_path / "reshape" / "identity"
    folder.mkdir(parents=True)
    shutil.copyfile(runfile(legacy_asm_path(kernel_spec("identity"))), folder / "identity.asm")
    (folder / "kernel.toml").write_text(TOML)
    (folder / "gen.py").write_text(GENERATOR.format(plant="pass"))
    return tmp_path


def _kernel(tree: Path) -> Path:
    return tree / "reshape" / "identity"


def _plant_input_lane(tree: Path) -> None:
    """Change one lane of the input: row 0, lane 5."""
    (_kernel(tree) / "gen.py").write_text(GENERATOR.format(plant="values[5] += 1"))


# --- compare_setup -----------------------------------------------------------------


@pytest.mark.parametrize("case", ["default", "single_row"])
def test_an_unchanged_kernel_has_no_setup_difference(tree, case):
    report = setup_report("identity", str(_kernel(tree)), case)
    assert report.identical, report.describe()
    assert report.xmem_bytes == (3 if case == "default" else 1) * 512
    assert report.describe().endswith("IDENTICAL")


def test_a_planted_wrong_abi_register_is_named(tree):
    # A register the old harness leaves at 0, programmed with a wrong value.
    toml = TOML.replace("[inputs.x]", '[derived]\ntail = "rows + 4"\n\n[inputs.x]')
    (_kernel(tree) / "kernel.toml").write_text(
        toml.replace('rows = "CR4"', 'rows = "CR4"\ntail = "CR7"')
    )
    report = setup_report("identity", str(_kernel(tree)))
    assert not report.identical
    assert report.registers == ("CR7: old 0, new 7",)
    assert "registers: 1 differ\n  CR7: old 0, new 7" in report.describe()


def test_a_planted_input_lane_is_located_by_region_row_and_lane(tree):
    _plant_input_lane(tree)
    report = setup_report("identity", str(_kernel(tree)))
    assert report.registers == ()
    assert report.xmem_total == 1
    assert report.xmem[0].startswith("x row 0 lane 5 (byte 20): old ")


def test_compare_setup_main_exits_by_the_verdict(tree, capsys):
    assert compare_setup.main(["identity", str(_kernel(tree))]) == 0
    _plant_input_lane(tree)
    assert compare_setup.main(["identity", str(_kernel(tree))]) == 1
    assert "DIFFERENT" in capsys.readouterr().out
    assert compare_setup.main(["identity", str(_kernel(tree)), "--case", "nope"]) == 2


# --- compare_legacy ----------------------------------------------------------------


def test_an_unchanged_kernel_is_identical_in_every_case(tree):
    rows = compare_family("reshape", only=["identity"], root=tree)
    assert [(row.case, row.passed) for row in rows] == [("default", True), ("single_row", True)]
    report = table(rows)
    assert "| identity | default | yes | yes | 17/17 |" in report
    assert "| identity | single_row | yes | yes | 7/7 |" in report
    assert "2 of 2 case(s) identical." in report


def test_a_planted_one_instruction_change_fails_the_binary(tree):
    asm = _kernel(tree) / "identity.asm"
    source = asm.read_text()
    assert source.count(IMMEDIATE) == 1
    asm.write_text(source.replace(IMMEDIATE, IMMEDIATE.replace(" 0 ", " 1 ")))
    rows = compare_family("reshape", only=["identity"], root=tree)
    assert all(not row.binary_ok and not row.passed for row in rows)
    assert any("is not the E02" in note for note in rows[0].notes)


def test_a_planted_one_lane_input_change_fails_the_output(tree):
    _plant_input_lane(tree)
    rows = compare_family("reshape", only=["identity"], root=tree)
    assert all(row.binary_ok and not row.output_ok for row in rows)
    assert "output y differs from the old XMEM" in " ".join(rows[0].notes)
    assert "| identity | default | yes | **NO** | 17/17 |" in table(rows)


def test_a_kernel_not_migrated_yet_is_a_failed_row(tree, capsys):
    shutil.rmtree(_kernel(tree))
    (tree / "reshape" / ".keep").write_text("")
    argv = ["reshape", "--kernel", "identity", "--kernels-root", str(tree)]
    assert compare_legacy.main(argv) == 1
    assert "no new kernel of this name" in capsys.readouterr().out


# --- a multi-stage kernel (E06-14) -------------------------------------------------

STAGES = """\
def run(ctx):
    for _ in range({times}):
        ctx.run_stage("identity.asm", abi={{"x": "CR2", "y": "CR3", "rows": "CR4"}})
"""


def _as_stages(tree: Path, times: int) -> None:
    """The identity kernel as a stages hook running its program ``times`` times."""
    toml = _kernel(tree) / "kernel.toml"
    toml.write_text(
        TOML.replace('formats = ["fp32"]', 'formats = ["fp32"]\nstages = "stages.py:run"')
    )
    (_kernel(tree) / "stages.py").write_text(STAGES.format(times=times))


def test_a_multi_stage_kernel_is_compared_by_its_stages_programs_and_cycles(tree):
    _as_stages(tree, 1)
    rows = compare_family("reshape", only=["identity"], root=tree)
    assert [(row.case, row.passed) for row in rows] == [("default", True), ("single_row", True)]
    assert "| identity | default | yes | yes | 17/17 |" in table(rows)


def test_a_multi_stage_kernel_running_more_than_the_old_harness_is_named(tree):
    # The same program twice is one distinct program, but twice the cycles.
    _as_stages(tree, 2)
    rows = compare_family("reshape", only=["identity"], root=tree)
    assert all(row.binary_ok and row.output_ok and not row.passed for row in rows)
    assert "cycles differ: old 17, new 34" in rows[0].notes


def test_compare_setup_refuses_a_multi_stage_kernel_and_names_the_other_tool(tree, capsys):
    _as_stages(tree, 1)
    assert compare_setup.main(["identity", str(_kernel(tree))]) == 2
    assert "compare_legacy compares its programs" in capsys.readouterr().err
