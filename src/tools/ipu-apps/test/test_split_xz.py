"""End-to-end regression tests for the split-to-x-and-z application.

Assemble → load → run → compare both outputs against golden references.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ipu_apps.split_xz import SplitXzApp


_INST_BIN = Path(os.environ["SPLIT_XZ_INST_BIN"])
_DATA_DIR = Path(os.environ["SPLIT_XZ_DATA_DIR"])


def _run_split(tmp_path: Path, stage: str) -> tuple[bytes, bytes, int]:
    """Run the split app for a stage, return (x_bytes, z_bytes, cycles)."""
    data_dir = _DATA_DIR / stage
    if not data_dir.exists():
        pytest.skip(f"Test data not found: {data_dir}")

    inputs = data_dir / "xz_in_int8.bin"
    if not inputs.exists():
        pytest.skip(f"Missing data file: {inputs}")

    x_out = tmp_path / "x.bin"
    z_out = tmp_path / "z.bin"
    app = SplitXzApp(
        inst_path=_INST_BIN,
        inputs_path=inputs,
        output_path=x_out,
        output_z_path=z_out,
        stage=stage,
    )
    _, cycles = app.run(max_cycles=2_000_000)
    return x_out.read_bytes(), z_out.read_bytes(), cycles


@pytest.mark.parametrize("stage", ["stage3", "stage4"])
def test_split(tmp_path: Path, stage: str) -> None:
    x_bytes, z_bytes, cycles = _run_split(tmp_path, stage)
    assert cycles > 0

    for actual, golden_name in ((x_bytes, "x_out_int8.bin"), (z_bytes, "z_out_int8.bin")):
        golden = _DATA_DIR / stage / golden_name
        if golden.exists():
            assert actual == golden.read_bytes(), f"{stage}/{golden_name} mismatch"


@pytest.mark.parametrize("stage", ["stage3", "stage4"])
def test_split_halves_reconstruct_the_input(tmp_path: Path, stage: str) -> None:
    """x and z are contiguous halves, so concatenating them is the input."""
    x_bytes, z_bytes, _ = _run_split(tmp_path, stage)
    inputs = (_DATA_DIR / stage / "xz_in_int8.bin").read_bytes()
    assert x_bytes + z_bytes == inputs
