"""Self-contained tests for the universal standard 3x3 stride-2 convolution
(cols=128, milestone 1).

Runtime-generates random INT8 weights and inputs, runs the emulator, and
compares against a numpy reference computed with ipu_mult/ipu_add for bit-
exact parity.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np
import pytest

from ipu_emu.ipu_math import DType, ipu_mult, ipu_add
from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.convolutions_universal.conv.conv_universal_stride import (
    ConvUniversalStrideApp,
    OUTPUT_BASE_ADDR,
    OUTPUT_CHUNK_BYTES,
)


ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "ipu_apps"
    / "convolutions_universal"
    / "conv" / "conv_universal_stride"
    / "conv_universal_stride.asm"
)

COLS = 128


def _pack_input_chunked(input_chw: np.ndarray, rows: int) -> bytes:
    """Pack [in_ch, rows, 128] int8 input: one spatial row per 128-byte chunk."""
    in_ch, ir, ic = input_chw.shape
    assert (ir, ic) == (rows, COLS)
    packed = bytearray(rows * in_ch * 128)
    for ch in range(in_ch):
        for r in range(rows):
            offset = (r * in_ch + ch) * 128
            packed[offset:offset + 128] = input_chw[ch, r, :].astype(np.uint8).tobytes()
    return bytes(packed)


def reference_conv_universal_stride(
    weights: np.ndarray,
    input_chw: np.ndarray,
    rows: int,
) -> bytes:
    """IPU-math-accurate stride-2/pad-1 reference.

    out_rows = rows // 2, out_cols = 64. Convolution centers are the even
    input rows/cols (0, 2, 4, ...). Two consecutive output rows are packed
    into one 128-byte chunk (rows_per_chunk_out = 2).
    """
    out_ch, in_ch, _, _ = weights.shape
    out_rows = rows // 2
    out_cols = COLS // 2
    rows_per_chunk_out = 2
    num_chunks = out_rows // rows_per_chunk_out
    output = bytearray(num_chunks * out_ch * 128)
    for f in range(out_ch):
        for orow in range(out_rows):
            r = orow * 2
            for ocol in range(out_cols):
                c = ocol * 2
                acc: int = 0
                for ic in range(in_ch):
                    for dr in range(3):
                        for dc in range(3):
                            nr, nc = r + dr - 1, c + dc - 1
                            if 0 <= nr < rows and 0 <= nc < COLS:
                                a = int(weights[f, ic, dr, dc])
                                b = int(input_chw[ic, nr, nc])
                                prod = ipu_mult(a, b, DType.INT8)
                                acc = ipu_add(acc, prod, DType.INT8)
                clamped = max(-128, min(127, acc))
                chunk = orow // rows_per_chunk_out
                local_row = orow % rows_per_chunk_out
                elem = local_row * out_cols + ocol
                out_idx = (chunk * out_ch + f) * 128 + elem
                output[out_idx] = clamped & 0xFF
    return bytes(output)


class TestConvUniversalStride:

    @pytest.fixture(scope="class")
    def inst_file(self, tmp_path_factory) -> Path:
        tmp = tmp_path_factory.mktemp("conv_universal_stride")
        inst_file = tmp / "conv_universal_stride.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(inst_file))
        return inst_file

    @pytest.mark.parametrize(
        "in_ch,out_ch,rows",
        [
            (1, 1, 4),         # smallest: 1 pair, 1 channel
            (16, 4, 4),        # partial last block, 1 pair
            (14, 4, 8),        # exactly one full block, 2 pairs
            (28, 4, 8),        # exactly two full blocks
            (10, 4, 8),        # single partial block
            (16, 8, 16),       # multiple pairs, multiple filters
        ],
    )
    def test_conv(
        self,
        inst_file: Path,
        tmp_path: Path,
        in_ch: int,
        out_ch: int,
        rows: int,
    ) -> None:
        rng = np.random.RandomState(42 + in_ch * 7 + out_ch + rows)
        weights = rng.randint(-32, 33, size=(out_ch, in_ch, 3, 3), dtype=np.int8)
        input_chw = rng.randint(-32, 33, size=(in_ch, rows, COLS), dtype=np.int8)

        input_packed = _pack_input_chunked(input_chw, rows)
        input_file = tmp_path / "input.bin"
        input_file.write_bytes(input_packed)

        app = ConvUniversalStrideApp(
            inst_path=inst_file,
            input_path=input_file,
            kernel=weights,
            output_path=None,
            dtype="INT8",
            rows=rows, cols=COLS,
            in_channels=in_ch, out_channels=out_ch,
        )

        num_pairs = (rows // 2) // 2
        max_cyc = 2_000 * max(num_pairs, 1) * 2 * out_ch * math.ceil(in_ch / 28) + 50_000
        state, cycles = app.run(max_cycles=max_cyc)
        assert cycles > 0

        total_bytes = num_pairs * out_ch * OUTPUT_CHUNK_BYTES
        actual = state.xmem.read_address(OUTPUT_BASE_ADDR, total_bytes)
        expected = reference_conv_universal_stride(weights, input_chw, rows)

        assert len(actual) == len(expected)

        mismatches = []
        out_cols = COLS // 2
        rows_per_chunk_out = 2
        for i in range(len(expected)):
            a_val = struct.unpack_from("b", actual, i)[0]
            e_val = struct.unpack_from("b", expected, i)[0]
            if a_val != e_val:
                elem = i % 128
                f_and_chunk = i // 128
                f = f_and_chunk % out_ch
                chunk = f_and_chunk // out_ch
                local_row = elem // out_cols
                col = elem % out_cols
                orow = chunk * rows_per_chunk_out + local_row
                mismatches.append(
                    f"  chunk={chunk} f={f} orow={orow} c={col} "
                    f"got={a_val} expected={e_val}"
                )
        assert not mismatches, (
            f"{len(mismatches)} mismatches (first 20):\n"
            + "\n".join(mismatches[:20])
        )
