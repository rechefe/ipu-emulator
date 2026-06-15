"""Self-contained tests for conv_universal_bn_activation (conv + folded bias + ReLU).

Derived from test_conv_universal. Runtime-generates random INT8 weights, inputs,
and per-filter biases, runs the emulator, and compares against a numpy reference
(ipu_mult/ipu_add for bit-exact parity) that seeds the accumulator with the bias,
applies ReLU, then clamps to INT8.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np
import pytest

from ipu_emu.ipu_math import DType, ipu_mult, ipu_add
from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.convolutions_universal.conv.conv_universal_bn_activation import (
    ConvUniversalBnActivationApp,
    OUTPUT_BASE_ADDR,
    OUTPUT_CHUNK_BYTES,
)


ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "ipu_apps"
    / "convolutions_universal"
    / "conv" / "conv_universal_bn_activation"
    / "conv_universal_bn_activation.asm"
)


def _pack_input_chunked(input_chw: np.ndarray, rows: int, cols: int) -> bytes:
    """Pack [in_ch, rows, cols] int8 input into the chunk-interleaved layout."""
    in_ch, ir, ic = input_chw.shape
    assert (ir, ic) == (rows, cols)
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


def reference_conv_bn_relu(
    weights: np.ndarray,
    input_chw: np.ndarray,
    bias: np.ndarray,
    rows: int,
    cols: int,
) -> bytes:
    """IPU-math-accurate reference: conv + per-filter bias seed + ReLU + INT8 clamp.

    The accumulator is seeded with the filter's bias (matching the asm's
    ``mult.ve.padded`` bias tap), accumulated with INT8 wrap semantics, then
    ReLU (max(0, .)) and clamped to [-128, 127] (== [0, 127] after ReLU).
    """
    out_ch, in_ch, _, _ = weights.shape
    rows_per_chunk = 128 // cols
    num_chunks = (rows * cols) // 128
    output = bytearray(num_chunks * out_ch * 128)
    for f in range(out_ch):
        for r in range(rows):
            for c in range(cols):
                acc: int = int(bias[f])  # bias seed
                for ic in range(in_ch):
                    for dr in range(3):
                        for dc in range(3):
                            nr, nc = r + dr - 1, c + dc - 1
                            if 0 <= nr < rows and 0 <= nc < cols:
                                a = int(weights[f, ic, dr, dc])
                                b = int(input_chw[ic, nr, nc])
                                prod = ipu_mult(a, b, DType.INT8)
                                acc = ipu_add(acc, prod, DType.INT8)
                acc = max(0, acc)               # ReLU
                clamped = max(-128, min(127, acc))
                chunk = r // rows_per_chunk
                local_row = r % rows_per_chunk
                elem = local_row * cols + c
                out_idx = (chunk * out_ch + f) * 128 + elem
                output[out_idx] = clamped & 0xFF
    return bytes(output)


class TestConvUniversalBnActivation:

    @pytest.fixture(scope="class")
    def inst_file(self, tmp_path_factory) -> Path:
        tmp = tmp_path_factory.mktemp("conv_universal_bn_activation")
        inst_file = tmp / "conv_universal_bn_activation.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(inst_file))
        return inst_file

    @pytest.mark.parametrize(
        "in_ch,out_ch,rows,cols",
        [
            (16, 4, 16, 16),   # partial last block (16 % 28 = 16)
            (14, 4, 16, 16),   # half a super-block
            (28, 4, 16, 16),   # exactly one full super-block (bias byte + 28*9 = 253)
            (10, 4, 16, 16),   # small partial block
            (16, 8, 32, 32),   # cross-chunk, multiple filters
            (40, 4, 16, 16),   # two super-blocks per filter (bias once)
        ],
    )
    def test_conv_bn_relu(
        self,
        inst_file: Path,
        tmp_path: Path,
        in_ch: int,
        out_ch: int,
        rows: int,
        cols: int,
    ) -> None:
        rng = np.random.RandomState(42 + in_ch * 7 + out_ch)
        weights = rng.randint(-32, 33, size=(out_ch, in_ch, 3, 3), dtype=np.int8)
        input_chw = rng.randint(-32, 33, size=(in_ch, rows, cols), dtype=np.int8)
        # Per-filter INT8 bias, spanning negative/positive so ReLU is exercised.
        bias = rng.randint(-80, 81, size=out_ch).astype(np.int8)

        input_packed = _pack_input_chunked(input_chw, rows, cols)
        input_file = tmp_path / "input.bin"
        input_file.write_bytes(input_packed)

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
        max_cyc = 2_000 * num_chunks * out_ch * math.ceil(in_ch / 28) + 50_000
        state, cycles = app.run(max_cycles=max_cyc)
        assert cycles > 0

        total_bytes = num_chunks * out_ch * OUTPUT_CHUNK_BYTES
        actual = state.xmem.read_address(OUTPUT_BASE_ADDR, total_bytes)
        expected = reference_conv_bn_relu(weights, input_chw, bias, rows, cols)

        assert len(actual) == len(expected)

        mismatches = []
        rows_per_chunk = 128 // cols
        for i in range(len(expected)):
            a_val = struct.unpack_from("b", actual, i)[0]
            e_val = struct.unpack_from("b", expected, i)[0]
            if a_val != e_val:
                elem = i % 128
                f_and_chunk = i // 128
                f = f_and_chunk % out_ch
                chunk = f_and_chunk // out_ch
                local_row = elem // cols
                col = elem % cols
                row = chunk * rows_per_chunk + local_row
                mismatches.append(
                    f"  chunk={chunk} f={f} r={row} c={col} "
                    f"got={a_val} expected={e_val}"
                )
        assert not mismatches, (
            f"{len(mismatches)} mismatches (first 20):\n"
            + "\n".join(mismatches[:20])
        )

    def test_relu_zeros_negative_outputs(self, inst_file: Path, tmp_path: Path) -> None:
        """A strongly negative bias with small weights must produce all-zero (ReLU) output."""
        rows = cols = 16
        in_ch, out_ch = 8, 2
        weights = np.zeros((out_ch, in_ch, 3, 3), dtype=np.int8)  # conv contributes 0
        input_chw = np.ones((in_ch, rows, cols), dtype=np.int8)
        bias = np.array([-50, -1], dtype=np.int8)  # negative -> ReLU clamps to 0

        input_file = tmp_path / "input.bin"
        input_file.write_bytes(_pack_input_chunked(input_chw, rows, cols))

        app = ConvUniversalBnActivationApp(
            inst_path=inst_file, input_path=input_file, kernel=weights, bias=bias,
            output_path=None, dtype="INT8", rows=rows, cols=cols,
            in_channels=in_ch, out_channels=out_ch,
        )
        num_chunks = (rows * cols) // 128
        state, _ = app.run(max_cycles=2_000 * num_chunks * out_ch + 50_000)
        out = state.xmem.read_address(OUTPUT_BASE_ADDR, num_chunks * out_ch * OUTPUT_CHUNK_BYTES)
        assert out == bytes(len(out)), "ReLU should zero all negative-bias outputs"
