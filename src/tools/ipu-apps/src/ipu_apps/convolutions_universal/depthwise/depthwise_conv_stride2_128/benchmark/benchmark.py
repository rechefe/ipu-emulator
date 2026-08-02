"""Benchmark depthwise_conv_stride2_128: cycles + real MULT util.

Two-stage app (unmodified depthwise_conv_universal at full resolution,
cols=128, followed by an ACC.STRIDE column-decimation pass). Reports cycles
and MULT-slot utilization read directly from ``state.stats`` (the emulator's
live per-cycle occupancy counter) -- the reported utilization is dominated by
stage 1 (the conv), since stage 2 is a short, taps-free decimation pass.

Usage::

    PYTHONPATH=... python -m ipu_apps.convolutions_universal.depthwise.depthwise_conv_stride2_128.benchmark.benchmark
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np

from ipu_emu.ipu_math import DType, ipu_mult, ipu_add

from ipu_apps.convolutions_universal.depthwise.depthwise_conv_stride2_128 import (
    DepthwiseConvStride2_128App,
    OUTPUT_BASE_ADDR,
    CHUNK_BYTES,
)
from ipu_apps.convolutions_universal.benchmarking import BenchRow, print_and_write_table


# (rows, channels) -- rows must be even and >=4, rows/2 (out_rows) must also
# be even (two output rows pack into one 128-byte chunk per stage 2 word).
#
# NOTE on the upper bound: the app's stage 1 (unmodified depthwise_conv_
# universal) writes its full-resolution output starting at row 9728
# (STAGE1_OUTPUT_BASE_ROW, byte 0x130000) and growing by `rows*channels`
# chunks; stage 2's own output region starts at OUTPUT_BASE_ADDR = 0x180000
# (row 12288). When rows*channels exceeds 12288-9728 = 2560, stage 1's tail
# chunks spill past 0x180000 and get clobbered by stage 2's own early writes
# (both stages share one IpuState/XMEM) before stage 2 gets a chance to read
# them -- a pre-existing memory-layout bug in the (untouched, per task
# constraints) app, not something introduced here. Configs below are kept at
# or under rows*channels == 2560 to avoid tripping it.
CONFIGS = [
    (32, 8),     # small spatial, few channels
    (64, 16),    # multi-chunk, mid channels
    (128, 16),   # large spatial, primary benchmark
    (32, 64),    # many channels, small spatial
]


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


def compare(actual: np.ndarray, expected: np.ndarray) -> int:
    return int(np.count_nonzero(actual != expected))


def run_config(rows: int, channels: int, seed: int):
    input_packed, kernel_raw, input_chw, kernel_ch9 = _gen_test_data(
        rows, channels, seed,
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        input_file = tmp / "input.bin"
        kernel_file = tmp / "kernel.bin"
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

        out_rows = rows // 2
        num_row_pairs = out_rows // 2

        expected = reference_stride2(input_chw, kernel_ch9, rows, channels)
        actual = np.zeros((channels, out_rows, 64), dtype=np.int8)
        for rp in range(num_row_pairs):
            for ch in range(channels):
                chunk_idx = rp * channels + ch
                chunk = state.xmem.read_address(
                    OUTPUT_BASE_ADDR + chunk_idx * CHUNK_BYTES, 128,
                )
                chunk_arr = np.frombuffer(chunk, dtype=np.int8)
                for local_row, orow in enumerate((2 * rp, 2 * rp + 1)):
                    actual[ch, orow, :] = chunk_arr[local_row * 64:(local_row + 1) * 64]

        # The app is two-stage: stage 1 (conv) and stage 2 (decimate) run as
        # separate emulator invocations that SHARE one IpuState, so
        # state.stats.mult_active_cycles accumulates across both stages
        # while state.stats.total_cycles gets overwritten with only stage
        # 2's cycle count (see run_until_complete in ipu_emu/emulator.py,
        # which does `state.stats.total_cycles = cycles` rather than +=).
        # Using state.stats.mult_utilization directly here would therefore
        # divide combined mult-active cycles by stage-2-only cycles,
        # producing bogus >100% figures. Recompute using the correct
        # combined total (cycles == cycles1 + cycles2, returned by
        # DepthwiseConvStride2_128App.run()) instead.
        mult_utilization = state.stats.mult_active_cycles / cycles if cycles else 0.0

    return cycles, compare(actual, expected), mult_utilization


def main() -> None:
    rows_out = []
    for rows, channels in CONFIGS:
        seed = 100 + rows * 7 + channels
        t0 = time.time()
        cycles, mismatches, mult_util = run_config(rows, channels, seed)
        elapsed = time.time() - t0
        rows_out.append(BenchRow(
            label=f"{rows}x{channels}",
            cycles=cycles,
            mult_utilization=mult_util,
            correct=(mismatches == 0),
            elapsed_s=elapsed,
        ))

    out_path = Path(__file__).resolve().parent / "results.md"
    print_and_write_table("depthwise_conv_stride2_128", rows_out, out_path)


if __name__ == "__main__":
    main()
