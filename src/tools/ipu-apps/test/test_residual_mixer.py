"""End-to-end regression tests for the residual_mixer application.

Assemble → load → run → compare output against golden reference.

The golden data is generated so that roughly a fifth of the sums fall outside
the INT8 range, so these tests also cover the clamp ACTIVATE.QUANTIZE applies
on the way out of R_ACC.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ipu_apps.residual_mixer import ResidualMixerApp


_INST_BIN = Path(os.environ["RESIDUAL_MIXER_INST_BIN"])
_DATA_DIR = Path(os.environ["RESIDUAL_MIXER_DATA_DIR"])


def _run_residual(tmp_path: Path, stage: str) -> tuple[bytes, int]:
    """Run the residual app for a stage, return (output_bytes, cycles)."""
    data_dir = _DATA_DIR / stage
    if not data_dir.exists():
        pytest.skip(f"Test data not found: {data_dir}")

    skip = data_dir / "skip_in_int8.bin"
    branch = data_dir / "branch_in_int8.bin"
    if not skip.exists() or not branch.exists():
        pytest.skip(f"Missing data files in {data_dir}")

    output = tmp_path / "out.bin"
    app = ResidualMixerApp(
        inst_path=_INST_BIN,
        inputs_path=skip,
        branch_path=branch,
        output_path=output,
        stage=stage,
    )
    _, cycles = app.run(max_cycles=2_000_000)
    return output.read_bytes(), cycles


@pytest.mark.parametrize("stage", ["stage3", "stage4"])
def test_residual_mixer(tmp_path: Path, stage: str) -> None:
    """x + gamma_1 * mixer(norm1(x)), summed at INT32 and clamped to INT8."""
    actual, cycles = _run_residual(tmp_path, stage)
    assert cycles > 0
    golden = _DATA_DIR / stage / "out_int8.bin"
    if golden.exists():
        assert actual == golden.read_bytes()


@pytest.mark.parametrize("stage", ["stage3", "stage4"])
def test_residual_mixer_exercises_the_int8_clamp(tmp_path: Path, stage: str) -> None:
    """The reference data is only meaningful if some sums saturate."""
    actual, _ = _run_residual(tmp_path, stage)
    saturated = sum(1 for b in actual if b in (0x7F, 0x80))
    assert saturated > 0
