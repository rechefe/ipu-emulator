"""Benchmark depthwise_conv_stride2_narrow: cycles + real MULT util.

Two-stage depthwise 3x3 stride-2 conv, cols in {16, 32, 64} (packed
multi-row-per-chunk layout). Reports cycles and MULT-slot utilization read
directly from ``state.stats`` (the emulator's live per-cycle occupancy
counter) for stage 1 + stage 2 combined.

Usage::

    PYTHONPATH=... python -m ipu_apps.convolutions_universal.depthwise.depthwise_conv_stride2_narrow.benchmark.benchmark
"""

from __future__ import annotations

import struct
import tempfile
import time
from pathlib import Path

import numpy as np

from ipu_emu.ipu_math import DType, ipu_mult, ipu_add

from ipu_apps.convolutions_universal.depthwise.depthwise_conv_stride2_narrow import (
    DepthwiseConvStride2NarrowApp,
    OUTPUT_BASE_ADDR,
    CHUNK_BYTES,
)
from ipu_apps.convolutions_universal.benchmarking import BenchRow, print_and_write_table


# (rows, cols, channels) — rows must be a multiple of 4*rows_per_chunk,
# rows_per_chunk = 128/cols: cols=64 -> mult of 8, cols=32 -> mult of 16,
# cols=16 -> mult of 32.
CONFIGS = [
    (8, 64, 2),        # cols=64, minimal
    (8, 64, 16),        # cols=64, more channels
    (16, 32, 4),        # cols=32, small
    (16, 32, 24),       # cols=32, larger channel count
    (32, 16, 3),        # cols=16, odd/non-packing-multiple channel count
    (32, 16, 16),        # cols=16, larger channel count
]


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


def run_config(rows: int, cols: int, channels: int, seed: int):
    input_packed, kernel_raw, input_chw, kernel_ch9 = _gen_test_data(
        rows, cols, channels, seed,
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        input_file = tmp / "input.bin"
        kernel_file = tmp / "kernel.bin"
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

        expected = reference_stride2_narrow(input_chw, kernel_ch9, rows, cols, channels)
        out_cols = cols // 2
        out_rows_per_chunk = 128 // out_cols
        num_out_groups = (rows // 2) // out_rows_per_chunk

        mismatches = 0
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
                            mismatches += 1

    return cycles, mismatches, state.stats.mult_utilization


def main() -> None:
    rows_out = []
    for i, (rows, cols, channels) in enumerate(CONFIGS):
        t0 = time.time()
        cycles, mismatches, mult_util = run_config(rows, cols, channels, seed=100 + i)
        elapsed = time.time() - t0
        rows_out.append(BenchRow(
            label=f"{rows}x{cols}x{channels}",
            cycles=cycles,
            mult_utilization=mult_util,
            correct=(mismatches == 0),
            elapsed_s=elapsed,
        ))

    out_path = Path(__file__).resolve().parent / "results.md"
    print_and_write_table("depthwise_conv_stride2_narrow", rows_out, out_path)


if __name__ == "__main__":
    main()
