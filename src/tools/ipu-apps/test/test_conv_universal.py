"""Self-contained tests for the universal standard 3x3 convolution (FPB=28, FP32).

Runtime-generates random FP32 weights and inputs, runs the emulator, and
compares against a real ``torch.nn.functional.conv2d`` reference
(tolerance-based, since IPU FP32 accumulation order differs from PyTorch's).
Exercises full/partial kernel blocks, cross-chunk spatial sizes, and
non-power-of-2/padding-heavy shapes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.convolutions_universal.conv.conv_universal import ConvUniversalApp

ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "ipu_apps"
    / "convolutions_universal"
    / "conv" / "conv_universal"
    / "conv_universal.asm"
)

_TOL = 1e-2


def reference_conv_universal(weights: np.ndarray, input_chw: np.ndarray) -> np.ndarray:
    """Real PyTorch reference: 3x3 conv, stride 1, "same" padding."""
    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(input_chw).unsqueeze(0)
    w = torch.from_numpy(weights)
    return F.conv2d(x, w, padding=1).squeeze(0).numpy()


class TestConvUniversal:

    @pytest.fixture(scope="class")
    def inst_file(self, tmp_path_factory) -> Path:
        tmp = tmp_path_factory.mktemp("conv_universal")
        inst_file = tmp / "conv_universal.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(inst_file))
        return inst_file

    @pytest.mark.parametrize(
        "in_ch,out_ch,height,width",
        [
            (16, 4, 16, 16),   # partial last block (16 % 28)
            (28, 4, 16, 16),   # exactly one full block
            (56, 4, 16, 16),   # exactly two full blocks
            (10, 4, 16, 16),   # single partial block (10 < 28)
            (16, 8, 32, 32),   # cross-chunk, multiple filters
            (4, 2, 8, 128),    # cols=128: one packed row per chunk (Partition.P0)
            (16, 4, 8, 128),   # cols=128, partial kernel block
            # Non-power-of-2 / padding-heavy shapes -- exercises internal
            # padding via next_valid_cols/min_rows_for_chunk_floor.
            (4, 2, 8, 8),
            (4, 2, 5, 5),
        ],
    )
    def test_conv(
        self,
        inst_file: Path,
        tmp_path: Path,
        in_ch: int,
        out_ch: int,
        height: int,
        width: int,
    ) -> None:
        rng = np.random.RandomState(42 + in_ch * 7 + out_ch)
        weights = (rng.randn(out_ch, in_ch, 3, 3) * 0.2).astype(np.float32)
        input_chw = (rng.randn(in_ch, height, width) * 0.5).astype(np.float32)

        input_file = tmp_path / "input.bin"
        input_file.write_bytes(input_chw.tobytes())

        app = ConvUniversalApp(
            inst_path=inst_file,
            input_path=input_file,
            kernel=weights,
            output_path=None,
            height=height, width=width,
            in_channels=in_ch, out_channels=out_ch,
        )

        max_cyc = 2_000 * app.num_chunks * out_ch * app.blocks_per_filter + 50_000
        state, cycles = app.run(max_cycles=max_cyc)
        assert cycles > 0

        total_elements = app.num_chunks * out_ch * 128
        raw = state.xmem.read_address(app.output_base_addr, total_elements * 4)
        from ipu_apps.convolutions_universal import unpack_output_chunked
        padded_out = unpack_output_chunked(raw, out_ch, app.rows, app.cols)
        actual = padded_out[:, :height, :width]
        expected = reference_conv_universal(weights, input_chw)

        diff = np.abs(actual - expected).max()
        assert diff < _TOL, (
            f"max diff {diff:.3e} for in_ch={in_ch} out_ch={out_ch} "
            f"{height}x{width}\n"
            f"  actual[0,0,:8]:   {actual[0, 0, :8]}\n"
            f"  expected[0,0,:8]: {expected[0, 0, :8]}"
        )
