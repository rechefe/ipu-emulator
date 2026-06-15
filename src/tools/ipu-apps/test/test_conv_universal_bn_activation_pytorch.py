"""Cross-validate conv_universal_bn_activation against a PyTorch INT8 reference.

The app computes a *raw integer* quantized conv: int8 x int8 -> int32 products,
int32 accumulation, + int8 per-filter bias, ReLU, then clamp to [-128, 127]
(no per-channel rescale; the requantization "scale" is effectively 1, zero-point
0). The matching PyTorch reference therefore runs ``F.conv2d`` on int-valued
float tensors (exact for these magnitudes), adds the bias, applies ReLU, and
clamps — NOT torch's scale/zero-point quantized ops, which model a different
(rescaled) pipeline the app does not implement.

This complements the numpy ipu-math oracle in test_conv_universal_bn_activation
by checking the convolution math against an independent framework.

Skipped automatically if torch is not installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from ipu_as.lark_tree import assemble_to_bin_file  # noqa: E402

from ipu_apps.convolutions_universal.conv.conv_universal_bn_activation import (  # noqa: E402
    ConvUniversalBnActivationApp,
    OUTPUT_BASE_ADDR,
    OUTPUT_CHUNK_BYTES,
)


ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "src" / "ipu_apps" / "convolutions_universal"
    / "conv" / "conv_universal_bn_activation"
    / "conv_universal_bn_activation.asm"
)


def _pack_input_chunked(input_chw: np.ndarray, rows: int, cols: int) -> bytes:
    in_ch = input_chw.shape[0]
    rows_per_chunk = 128 // cols
    num_chunks = (rows * cols) // 128
    packed = bytearray(num_chunks * in_ch * 128)
    for ch in range(in_ch):
        for r in range(rows):
            for c in range(cols):
                chunk = r // rows_per_chunk
                local_row = r % rows_per_chunk
                offset = (chunk * in_ch + ch) * 128 + local_row * cols + c
                packed[offset] = np.uint8(input_chw[ch, r, c]).item()
    return bytes(packed)


def pytorch_int8_conv_bn_relu(
    weights: np.ndarray,    # [out_ch, in_ch, 3, 3] int8
    input_chw: np.ndarray,  # [in_ch, rows, cols]   int8
    bias: np.ndarray,       # [out_ch]              int8
    rows: int,
    cols: int,
) -> bytes:
    """PyTorch INT8 conv + folded bias + ReLU + clamp -> chunk-interleaved bytes.

    int8 x int8 -> int32 conv (scale 1, zero-point 0); int32 bias; ReLU; clamp
    to [-128, 127]. Uses int64 tensors so the conv is exact (no float rounding).
    """
    out_ch = weights.shape[0]
    rows_per_chunk = 128 // cols
    num_chunks = (rows * cols) // 128

    w = torch.from_numpy(weights.astype(np.int64))          # [oc, ic, 3, 3]
    x = torch.from_numpy(input_chw.astype(np.int64))[None]  # [1, ic, rows, cols]

    # Exact integer convolution via int64; zero-padding of 1 == the app's borders.
    y = F.conv2d(x, w, bias=None, stride=1, padding=1)      # [1, oc, rows, cols]
    y = y[0]
    y = y + torch.from_numpy(bias.astype(np.int64))[:, None, None]  # folded bias
    y = torch.clamp(y, min=0)                                # ReLU
    y = torch.clamp(y, -128, 127).to(torch.int8).numpy()

    output = bytearray(num_chunks * out_ch * 128)
    for f in range(out_ch):
        for r in range(rows):
            chunk = r // rows_per_chunk
            local_row = r % rows_per_chunk
            for c in range(cols):
                elem = local_row * cols + c
                out_idx = (chunk * out_ch + f) * 128 + elem
                output[out_idx] = np.uint8(y[f, r, c]).item()
    return bytes(output)


@pytest.mark.parametrize(
    "in_ch,out_ch,rows,cols",
    [
        (16, 4, 16, 16),
        (14, 4, 16, 16),
        (28, 4, 16, 16),
        (40, 4, 16, 16),
        (16, 8, 32, 32),
        (32, 16, 32, 32),
    ],
)
def test_matches_pytorch_int8(tmp_path: Path, in_ch, out_ch, rows, cols) -> None:
    rng = np.random.RandomState(1234 + in_ch * 7 + out_ch * 13 + rows + cols)
    weights = rng.randint(-4, 5, size=(out_ch, in_ch, 3, 3), dtype=np.int8)
    input_chw = rng.randint(-8, 9, size=(in_ch, rows, cols), dtype=np.int8)
    bias = rng.randint(-80, 81, size=out_ch).astype(np.int8)

    inst_file = tmp_path / "conv_universal_bn_activation.bin"
    assemble_to_bin_file(ASM_PATH.read_text(), str(inst_file))

    input_file = tmp_path / "input.bin"
    input_file.write_bytes(_pack_input_chunked(input_chw, rows, cols))

    app = ConvUniversalBnActivationApp(
        inst_path=inst_file,
        input_path=input_file,
        kernel=weights,
        bias=bias,
        output_path=None,
        dtype="INT8",
        rows=rows, cols=cols,
        in_channels=in_ch, out_channels=out_ch,
    )

    num_chunks = (rows * cols) // 128
    state, cycles = app.run(max_cycles=2_000_000)
    assert cycles > 0

    total_bytes = num_chunks * out_ch * OUTPUT_CHUNK_BYTES
    actual = state.xmem.read_address(OUTPUT_BASE_ADDR, total_bytes)
    expected = pytorch_int8_conv_bn_relu(weights, input_chw, bias, rows, cols)

    # Compare as signed int8.
    a = np.frombuffer(actual, dtype=np.int8)
    e = np.frombuffer(expected, dtype=np.int8)
    n_mismatch = int(np.count_nonzero(a != e))
    assert n_mismatch == 0, (
        f"{n_mismatch}/{len(e)} bytes differ from PyTorch INT8 reference "
        f"(in_ch={in_ch} out_ch={out_ch} {rows}x{cols})"
    )
