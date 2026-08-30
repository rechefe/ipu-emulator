"""Chains the MambaVision kernels the way ``Block.forward`` sequences them.

The per-kernel tests prove each program against a golden file. This one proves
they *compose*: it runs them back to back on real intermediate buffers, exactly
as the RISC core would schedule them for one block, and checks two properties
no single kernel can check on its own.

1.  ``reshape → split → concat → reshape`` is the identity. In
    ``MambaVisionMixer.forward`` the tensor goes token view → channel view, is
    chunked into x and z, and after the scan is concatenated back and returned
    to token view. Replace the scan with a pass-through (y = x) and the whole
    data-movement path must give the input back byte for byte. Any layout,
    padding or stride mistake in the three kernels breaks this.

2.  The two residuals chain. The MLP residual's skip input is the mixer
    residual's output, which is how ``Block.forward`` is written; running them
    in sequence checks that one kernel's packed output buffer really is a
    valid input buffer for the next.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ipu_apps.concat_yz import ConcatYzApp
from ipu_apps.reshape_token_view import (
    ROW_BYTES,
    ReshapeTokenViewApp,
    pad_to_row,
    parse_stage,
    rows_for,
)
from ipu_apps.residual_mixer import ResidualMixerApp
from ipu_apps.residual_mlp import ResidualMlpApp
from ipu_apps.split_xz import SplitXzApp


_RESHAPE_BIN = Path(os.environ["RESHAPE_INST_BIN"])
_SPLIT_BIN = Path(os.environ["SPLIT_XZ_INST_BIN"])
_CONCAT_BIN = Path(os.environ["CONCAT_YZ_INST_BIN"])
_RESIDUAL_MIXER_BIN = Path(os.environ["RESIDUAL_MIXER_INST_BIN"])
_RESIDUAL_MLP_BIN = Path(os.environ["RESIDUAL_MLP_INST_BIN"])
_RESHAPE_DATA_DIR = Path(os.environ["RESHAPE_DATA_DIR"])
_RESIDUAL_MIXER_DATA_DIR = Path(os.environ["RESIDUAL_MIXER_DATA_DIR"])

MAX_CYCLES = 20_000_000


@pytest.mark.parametrize("stage", ["stage3", "stage4"])
def test_mixer_data_movement_round_trips(tmp_path: Path, stage: str) -> None:
    d_model, seq_len = parse_stage(stage)
    d_inner = d_model  # expand == 1

    xz_in = _RESHAPE_DATA_DIR / stage / "t2c_in_int8.bin"
    if not xz_in.exists():
        pytest.skip(f"Test data not found: {xz_in}")

    # 1. rearrange(xz, "b l d -> b d l")
    channel_view = tmp_path / "channel_view.bin"
    ReshapeTokenViewApp(
        inst_path=_RESHAPE_BIN,
        inputs_path=xz_in,
        output_path=channel_view,
        stage=stage,
        direction="t2c",
    ).run(max_cycles=MAX_CYCLES)

    # 2. x, z = xz.chunk(2, dim=1)
    x_out = tmp_path / "x.bin"
    z_out = tmp_path / "z.bin"
    SplitXzApp(
        inst_path=_SPLIT_BIN,
        inputs_path=channel_view,
        output_path=x_out,
        output_z_path=z_out,
        stage=stage,
    ).run(max_cycles=MAX_CYCLES)

    # 3. the selective scan sits here; stand in for it with y = x so the round
    #    trip stays exact and only the data movement is under test.
    y_out = x_out

    # 4. y = torch.cat([y, z], dim=1)
    yz_out = tmp_path / "yz.bin"
    ConcatYzApp(
        inst_path=_CONCAT_BIN,
        inputs_path=y_out,
        z_path=z_out,
        output_path=yz_out,
        stage=stage,
    ).run(max_cycles=MAX_CYCLES)

    # 5. rearrange(y, "b d l -> b l d"). The transpose reads a whole 128-line
    #    block per output row, so the source is padded up to a 128-line
    #    boundary first -- on hardware that is the DMA descriptor's job.
    pad_lines = pad_to_row(d_inner) - d_inner
    padded = tmp_path / "yz_padded.bin"
    padded.write_bytes(
        yz_out.read_bytes() + bytes(pad_lines * rows_for(seq_len) * ROW_BYTES)
    )

    back = tmp_path / "back.bin"
    ReshapeTokenViewApp(
        inst_path=_RESHAPE_BIN,
        inputs_path=padded,
        output_path=back,
        stage=stage,
        direction="c2t",
    ).run(max_cycles=MAX_CYCLES)

    original = xz_in.read_bytes()
    result = back.read_bytes()
    assert result == original[: len(result)]


@pytest.mark.parametrize("stage", ["stage3", "stage4"])
def test_the_two_residuals_chain(tmp_path: Path, stage: str) -> None:
    data_dir = _RESIDUAL_MIXER_DATA_DIR / stage
    skip = data_dir / "skip_in_int8.bin"
    branch = data_dir / "branch_in_int8.bin"
    if not skip.exists() or not branch.exists():
        pytest.skip(f"Missing data files in {data_dir}")

    # x = x + gamma_1 * mixer(norm1(x))
    after_mixer = tmp_path / "after_mixer.bin"
    ResidualMixerApp(
        inst_path=_RESIDUAL_MIXER_BIN,
        inputs_path=skip,
        branch_path=branch,
        output_path=after_mixer,
        stage=stage,
    ).run(max_cycles=MAX_CYCLES)

    golden = data_dir / "out_int8.bin"
    if golden.exists():
        assert after_mixer.read_bytes() == golden.read_bytes()

    # x = x + gamma_2 * mlp(norm2(x)), skipping from the previous result
    final = tmp_path / "final.bin"
    ResidualMlpApp(
        inst_path=_RESIDUAL_MLP_BIN,
        inputs_path=after_mixer,
        branch_path=branch,
        output_path=final,
        stage=stage,
    ).run(max_cycles=MAX_CYCLES)

    # Same shape in, same shape out, and the second add really ran.
    assert len(final.read_bytes()) == len(after_mixer.read_bytes())
    assert final.read_bytes() != after_mixer.read_bytes()
