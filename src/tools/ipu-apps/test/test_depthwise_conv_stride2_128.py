"""Self-contained tests for depthwise 3x3 stride-2 conv, cols=128 (two-stage app).

Verifies the two-stage pipeline (unmodified depthwise_conv_universal at full
resolution + a decimate pass) matches an ipu_math reference: depthwise conv
(zero-pad) + clamp, output row/col stride 2, no bias, no ReLU.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from ipu_emu.ipu_math import DType, ipu_mult, ipu_add
from ipu_apps.convolutions_universal.depthwise.depthwise_conv_stride2_128 import (
    DepthwiseConvStride2_128App,
    OUTPUT_BASE_ADDR,
    CHUNK_BYTES,
)


def reference_stride2(
    input_chw: np.ndarray, kernel_ch9: np.ndarray, rows: int, channels: int,
) -> np.ndarray:
    """Depthwise 3x3 conv (zero-pad), stride 2 in both row and column.

    input_chw: [channels, rows, 128] int8. kernel_ch9: [channels, 9] int8
    (taps ordered dr*3+dc, dr/dc in -1..1). Returns [channels, rows//2, 64].
    """
    out_rows = rows // 2
    out_cols = 64
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
                        if 0 <= ir < rows and 0 <= ic < 128:
                            a = int(kernel_ch9[ch, dr * 3 + dc])
                            b = int(input_chw[ch, ir, ic])
                            prod = ipu_mult(a, b, dtype)
                            acc = ipu_add(acc, prod, dtype)
                out[ch, orow, ocol] = max(-128, min(127, acc))
    return out


def _gen_test_data(
    rows: int, channels: int, seed: int,
) -> tuple[bytes, bytes, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    input_chw = rng.randint(-4, 5, size=(channels, rows, 128)).astype(np.int8)
    kernel_ch9 = rng.randint(-4, 5, size=(channels, 9)).astype(np.int8)

    # Row-interleaved by channel: (row r, ch) chunk at (r*channels + ch)*128.
    packed = bytearray(rows * channels * 128)
    for r in range(rows):
        for ch in range(channels):
            off = (r * channels + ch) * 128
            packed[off:off + 128] = input_chw[ch, r, :].tobytes()

    return bytes(packed), kernel_ch9.tobytes(), input_chw, kernel_ch9


class TestDepthwiseConvStride2_128:

    @pytest.mark.parametrize(
        "rows,channels,seed",
        [
            (8, 2, 3),     # minimal case
            (16, 4, 7),    # exercises the ch_loop cross-word row advance twice
            (32, 3, 11),   # odd channel count, larger spatial extent
        ],
    )
    def test_stride2(
        self, tmp_path: Path, rows: int, channels: int, seed: int,
    ) -> None:
        input_packed, kernel_raw, input_chw, kernel_ch9 = _gen_test_data(
            rows, channels, seed,
        )

        input_file = tmp_path / "input.bin"
        kernel_file = tmp_path / "kernel.bin"
        input_file.write_bytes(input_packed)
        kernel_file.write_bytes(kernel_raw)

        app = DepthwiseConvStride2_128App(
            input_path=input_file,
            kernel_path=kernel_file,
            output_path=None,
            rows=rows,
            channels=channels,
        )
        state, cycles = app.run(max_cycles=2_000_000)
        assert cycles > 0

        expected = reference_stride2(input_chw, kernel_ch9, rows, channels)
        out_rows = rows // 2
        num_row_pairs = out_rows // 2

        mismatches = []
        for rp in range(num_row_pairs):
            for ch in range(channels):
                chunk_idx = rp * channels + ch
                actual = state.xmem.read_address(
                    OUTPUT_BASE_ADDR + chunk_idx * CHUNK_BYTES, 128,
                )
                for local_row, orow in enumerate((2 * rp, 2 * rp + 1)):
                    for c in range(64):
                        a_val = struct.unpack_from("b", actual, local_row * 64 + c)[0]
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
            DepthwiseConvStride2_128App(
                input_path=input_file, kernel_path=kernel_file,
                output_path=None, rows=5, channels=1,
            )
