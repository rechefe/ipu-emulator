"""End-to-end regression tests for the concat-y-and-z application.

Assemble → load → run → compare output against golden reference.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ipu_apps.concat_yz import ConcatYzApp


_INST_BIN = Path(os.environ["CONCAT_YZ_INST_BIN"])
_DATA_DIR = Path(os.environ["CONCAT_YZ_DATA_DIR"])


def _run_concat(tmp_path: Path, stage: str) -> tuple[bytes, int]:
    """Run the concat app for a stage, return (output_bytes, cycles)."""
    data_dir = _DATA_DIR / stage
    if not data_dir.exists():
        pytest.skip(f"Test data not found: {data_dir}")

    y_in = data_dir / "y_in_int8.bin"
    z_in = data_dir / "z_in_int8.bin"
    if not y_in.exists() or not z_in.exists():
        pytest.skip(f"Missing data files in {data_dir}")

    output = tmp_path / "yz.bin"
    app = ConcatYzApp(
        inst_path=_INST_BIN,
        inputs_path=y_in,
        z_path=z_in,
        output_path=output,
        stage=stage,
    )
    _, cycles = app.run(max_cycles=2_000_000)
    return output.read_bytes(), cycles


@pytest.mark.parametrize("stage", ["stage3", "stage4"])
def test_concat(tmp_path: Path, stage: str) -> None:
    actual, cycles = _run_concat(tmp_path, stage)
    assert cycles > 0
    golden = _DATA_DIR / stage / "yz_out_int8.bin"
    if golden.exists():
        assert actual == golden.read_bytes()


@pytest.mark.parametrize("stage", ["stage3", "stage4"])
def test_concat_stacks_the_two_inputs(tmp_path: Path, stage: str) -> None:
    """cat along the channel axis is y's rows followed by z's rows."""
    actual, _ = _run_concat(tmp_path, stage)
    data_dir = _DATA_DIR / stage
    y_bytes = (data_dir / "y_in_int8.bin").read_bytes()
    z_bytes = (data_dir / "z_in_int8.bin").read_bytes()
    assert actual == y_bytes + z_bytes
