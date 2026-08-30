"""End-to-end regression tests for the reshape-to-token-view application.

Assemble → load → run → compare output against golden reference.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ipu_apps.reshape_token_view import (
    ROW_BYTES,
    ReshapeTokenViewApp,
    pad_to_row,
    parse_stage,
    rows_for,
)


_INST_BIN = Path(os.environ["RESHAPE_INST_BIN"])
_DATA_DIR = Path(os.environ["RESHAPE_DATA_DIR"])


def _run_reshape(
    tmp_path: Path, stage: str, direction: str, inputs: Path | None = None
) -> tuple[bytes, int]:
    """Run the reshape app for a stage and direction, return (output_bytes, cycles)."""
    data_dir = _DATA_DIR / stage
    if not data_dir.exists():
        pytest.skip(f"Test data not found: {data_dir}")

    if inputs is None:
        inputs = data_dir / f"{direction}_in_int8.bin"
    if not inputs.exists():
        pytest.skip(f"Missing data file: {inputs}")

    output = tmp_path / f"{stage}_{direction}_out.bin"
    app = ReshapeTokenViewApp(
        inst_path=_INST_BIN,
        inputs_path=inputs,
        output_path=output,
        stage=stage,
        direction=direction,
    )
    _, cycles = app.run(max_cycles=20_000_000)
    return output.read_bytes(), cycles


@pytest.mark.parametrize("stage", ["stage3", "stage4"])
@pytest.mark.parametrize("direction,golden_name", [
    ("t2c", "t2c_out_int8.bin"),
    ("c2t", "c2t_out_int8.bin"),
])
def test_reshape(tmp_path: Path, stage: str, direction: str, golden_name: str) -> None:
    actual, cycles = _run_reshape(tmp_path, stage, direction)
    assert cycles > 0
    golden = _DATA_DIR / stage / golden_name
    if golden.exists():
        assert actual == golden.read_bytes()


@pytest.mark.parametrize("stage", ["stage3", "stage4"])
def test_reshape_round_trip(tmp_path: Path, stage: str) -> None:
    """token view → channel view → token view must reproduce the input.

    Needs no golden data of its own, so it stays valid if the references are
    ever regenerated with different values.
    """
    d_model, seq_len = parse_stage(stage)
    d_inner = d_model  # expand == 1

    # forward: (L, d_inner) -> (d_inner, L)
    channel_view, _ = _run_reshape(tmp_path, stage, "t2c")

    # The transpose reads a whole 128-line block per output row, so the
    # intermediate has to be padded up to a 128-line boundary before it can
    # go back the other way. On hardware that padding is the DMA's job.
    pad_lines = pad_to_row(d_inner) - d_inner
    padded = tmp_path / "channel_view_padded.bin"
    padded.write_bytes(
        channel_view + bytes(pad_lines * rows_for(seq_len) * ROW_BYTES)
    )

    back, _ = _run_reshape(tmp_path, stage, "c2t", inputs=padded)

    original = (_DATA_DIR / stage / "t2c_in_int8.bin").read_bytes()
    assert back == original[: len(back)]
