"""Self-contained tests for depthwise 3x3 stride-2 conv, cols in {16,32,64}
(packed-row two-stage app).

Verifies the two-stage pipeline (unmodified depthwise_conv_universal at full
resolution + a joint row/col decimate pass) matches a numpy reference:
depthwise conv (zero-pad) + clamp, output row/col stride 2, no bias, no ReLU,
in the packed multi-row-per-chunk layout depthwise_conv_universal uses.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from ipu_emu.ipu_math import DType, ipu_mult, ipu_add
from ipu_apps.convolutions_universal.depthwise.depthwise_conv_stride2_narrow import (
    DepthwiseConvStride2NarrowApp,
    OUTPUT_BASE_ADDR,
    CHUNK_BYTES,
)


def reference_stride2_narrow(
    input_chw: np.ndarray, kernel_ch9: np.ndarray, rows: int, cols: int, channels: int,
) -> np.ndarray:
    """Depthwise 3x3 conv (zero-pad), stride 2 in both row and column.

    input_chw: [channels, rows, cols] int8. kernel_ch9: [channels, 9] int8
    (taps ordered dr*3+dc, dr/dc in -1..1). Returns [channels, rows//2, cols//2].
    """
    out_rows = rows // 2
    out_cols = cols // 2
    dtype = DType.INT8
    out = np.zeros((channels, out_rows, out_cols), dtype=np.int8)
    for ch in range(channels):
        for orow in range(out_rows):
            r_center = 2 * orow
            for ocol in range(out_cols):
                c_center = 2 * ocol
                acc = 0
                for dr in range(3):
                    for dc in range(3):
                        ir = r_center + dr - 1
                        ic = c_center + dc - 1
                        if 0 <= ir < rows and 0 <= ic < cols:
                            a = int(kernel_ch9[ch, dr * 3 + dc])
                            b = int(input_chw[ch, ir, ic])
                            prod = ipu_mult(a, b, dtype)
                            acc = ipu_add(acc, prod, dtype)
                out[ch, orow, ocol] = max(-128, min(127, acc))
    return out


def _pack_chw(input_chw: np.ndarray, rows: int, cols: int, channels: int) -> bytes:
    """Pack [channels, rows, cols] int8 into depthwise_conv_universal's
    chunk-interleaved layout: row-group g = r // rows_per_chunk holds
    `channels` 128-byte chunks (one per channel); within a chunk, local row
    `r % rows_per_chunk` occupies bytes [local_row*cols : local_row*cols+cols).
    """
    rows_per_chunk = 128 // cols
    num_row_groups = rows // rows_per_chunk
    packed = bytearray(num_row_groups * channels * 128)
    for ch in range(channels):
        for r in range(rows):
            g = r // rows_per_chunk
            local_row = r % rows_per_chunk
            off = (g * channels + ch) * 128 + local_row * cols
            packed[off:off + cols] = input_chw[ch, r, :].tobytes()
    return bytes(packed)


def _gen_test_data(
    rows: int, cols: int, channels: int, seed: int,
) -> tuple[bytes, bytes, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    input_chw = rng.randint(-4, 5, size=(channels, rows, cols)).astype(np.int8)
    kernel_ch9 = rng.randint(-4, 5, size=(channels, 9)).astype(np.int8)

    packed = _pack_chw(input_chw, rows, cols, channels)
    return packed, kernel_ch9.tobytes(), input_chw, kernel_ch9


class TestDepthwiseConvStride2Narrow:

    @pytest.mark.parametrize(
        "cols,rows,channels,seed",
        [
            (64, 8, 2, 3),     # rows_per_chunk=2, minimal
            (32, 16, 4, 7),    # rows_per_chunk=4
            (16, 32, 3, 11),   # rows_per_chunk=8, odd channel count
        ],
    )
    def test_stride2_narrow(
        self, tmp_path: Path, cols: int, rows: int, channels: int, seed: int,
    ) -> None:
        input_packed, kernel_raw, input_chw, kernel_ch9 = _gen_test_data(
            rows, cols, channels, seed,
        )

        input_file = tmp_path / "input.bin"
        kernel_file = tmp_path / "kernel.bin"
        input_file.write_bytes(input_packed)
        kernel_file.write_bytes(kernel_raw)

        app = DepthwiseConvStride2NarrowApp(
            input_path=input_file,
            kernel_path=kernel_file,
            output_path=None,
            rows=rows,
            cols=cols,
            channels=channels,
        )
        state, cycles = app.run(max_cycles=2_000_000)
        assert cycles > 0

        expected = reference_stride2_narrow(input_chw, kernel_ch9, rows, cols, channels)
        out_cols = cols // 2
        out_rows_per_chunk = 128 // out_cols
        num_out_groups = (rows // 2) // out_rows_per_chunk

        mismatches = []
        for og in range(num_out_groups):
            for ch in range(channels):
                chunk_idx = og * channels + ch
                actual = state.xmem.read_address(
                    OUTPUT_BASE_ADDR + chunk_idx * CHUNK_BYTES, 128,
                )
                for local_row in range(out_rows_per_chunk):
                    orow = og * out_rows_per_chunk + local_row
                    for c in range(out_cols):
                        a_val = struct.unpack_from(
                            "b", actual, local_row * out_cols + c,
                        )[0]
                        e_val = int(expected[ch, orow, c])
                        if a_val != e_val:
                            mismatches.append(
                                f"  ch={ch} orow={orow} col={c} got={a_val} expected={e_val}"
                            )
        assert not mismatches, (
            f"{len(mismatches)} mismatches (first 20):\n" + "\n".join(mismatches[:20])
        )

    def test_rejects_odd_rows(self, tmp_path: Path) -> None:
        input_file = tmp_path / "input.bin"
        kernel_file = tmp_path / "kernel.bin"
        input_file.write_bytes(b"\x00" * 128)
        kernel_file.write_bytes(b"\x00" * 9)
        with pytest.raises(ValueError):
            DepthwiseConvStride2NarrowApp(
                input_path=input_file, kernel_path=kernel_file,
                output_path=None, rows=5, cols=64, channels=1,
            )

    def test_rejects_non_multiple_of_4_row_groups(self, tmp_path: Path) -> None:
        input_file = tmp_path / "input.bin"
        kernel_file = tmp_path / "kernel.bin"
        input_file.write_bytes(b"\x00" * 128)
        kernel_file.write_bytes(b"\x00" * 9)
        # cols=16 -> rows_per_chunk=8 -> needs rows multiple of 32; 16 fails.
        with pytest.raises(ValueError):
            DepthwiseConvStride2NarrowApp(
                input_path=input_file, kernel_path=kernel_file,
                output_path=None, rows=16, cols=16, channels=1,
            )

    def test_rejects_invalid_cols(self, tmp_path: Path) -> None:
        input_file = tmp_path / "input.bin"
        kernel_file = tmp_path / "kernel.bin"
        input_file.write_bytes(b"\x00" * 128)
        kernel_file.write_bytes(b"\x00" * 9)
        with pytest.raises(ValueError):
            DepthwiseConvStride2NarrowApp(
                input_path=input_file, kernel_path=kernel_file,
                output_path=None, rows=8, cols=128, channels=1,
            )
