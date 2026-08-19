"""Benchmark pointwise_conv_unified across configs: cycles + correctness + efficiency.

Theoretical floor per output position: in_channels mults at 1 MAC/cyc.
Per row-group of 128 spatial positions: in_ch * out_ch mults across the
whole conv. Total mults = rows * cols * in_ch * out_ch / 128.

Usage::

    PYTHONPATH=... python -m ipu_apps.convolutions_universal.pointwise.pointwise_conv_unified.benchmark.benchmark
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.convolutions_universal.pointwise.pointwise_conv_unified import (
    PointwiseConvUnifiedApp,
)
from ipu_apps.convolutions_universal.benchmarking import BenchRow, print_and_write_table


ASM_PATH = (
    Path(__file__).resolve().parents[1] / "pointwise_conv_unified.asm"
)

# (rows, cols, in_ch, out_ch). Sized to fit XMEM (2 MB).
CONFIGS = [
    # Mirror the configs from pointwise_conv_universal benchmark for comparison
    (128, 128,  64,  32),
    (128, 128,  32,  48),
    ( 64,  64, 128,  64),
    ( 64,  64,  64, 128),
    ( 32,  32, 256,  96),
    ( 32,  32,  96, 256),
    ( 16,  16, 384, 128),
    ( 16,  16, 128, 256),
]


def pack_input(input_chw, rows, cols):
    channels, _, _ = input_chw.shape
    rows_per_chunk = 128 // cols
    row_groups = (rows * cols) // 128
    packed = bytearray(row_groups * channels * 128)
    for rg in range(row_groups):
        for ch in range(channels):
            dst = (rg * channels + ch) * 128
            for r in range(rows_per_chunk):
                spatial_row = rg * rows_per_chunk + r
                packed[dst + r * cols:dst + r * cols + cols] = (
                    input_chw[ch, spatial_row, :].view(np.uint8).tobytes()
                )
    return bytes(packed)


def reference_pointwise(weights, input_chw):
    inp32 = input_chw.astype(np.int32)
    w32 = weights.astype(np.int32)
    result = np.einsum("oi,ihw->ohw", w32, inp32)
    return result.clip(-128, 127).astype(np.int8)


def read_output(state, rows, cols, out_ch, output_base_addr):
    rows_per_chunk = 128 // cols
    row_groups = (rows * cols) // 128
    raw = state.xmem.read_address(output_base_addr, row_groups * out_ch * 128)
    vals = np.frombuffer(raw, dtype=np.uint8).reshape(row_groups, out_ch, 128)
    result = np.empty((out_ch, rows, cols), dtype=np.int8)
    for rg in range(row_groups):
        for oc in range(out_ch):
            block = vals[rg, oc, :rows_per_chunk * cols].view(np.int8)
            result[oc, rg * rows_per_chunk:(rg + 1) * rows_per_chunk, :] = (
                block.reshape(rows_per_chunk, cols)
            )
    return result


def run_config(inst_file, rows, cols, in_ch, out_ch):
    rng = np.random.RandomState(42 + in_ch * 7 + out_ch + rows + cols)
    weights = rng.randint(-3, 4, size=(out_ch, in_ch), dtype=np.int8)
    input_chw = rng.randint(-3, 4, size=(in_ch, rows, cols), dtype=np.int8)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        input_file = tmp / "input.bin"
        kernel_file = tmp / "kernel.bin"
        input_file.write_bytes(pack_input(input_chw, rows, cols))
        kernel_file.write_bytes(
            weights.reshape(out_ch * in_ch).view(np.uint8).tobytes()
        )

        app = PointwiseConvUnifiedApp(
            inst_path=inst_file,
            input_path=input_file,
            kernel_path=kernel_file,
            output_path=None,
            dtype="INT8",
            rows=rows, cols=cols, in_channels=in_ch, out_channels=out_ch,
        )
        max_cyc = 200 * in_ch * out_ch * (rows * cols // 128) + 100_000
        state, cycles = app.run(max_cycles=max_cyc)

        actual = read_output(state, rows, cols, out_ch, app.output_base_addr)
        expected = reference_pointwise(weights, input_chw)

    mismatches = int(np.sum(actual != expected))
    return cycles, mismatches, state.stats.mult_utilization


def main() -> None:
    src = ASM_PATH.read_text()
    with tempfile.TemporaryDirectory() as tmp:
        inst_file = Path(tmp) / "unified.bin"
        print("Assembling pointwise_conv_unified.asm ...", flush=True)
        assemble_to_bin_file(src, str(inst_file))

        rows_out = []
        for rows, cols, in_ch, out_ch in CONFIGS:
            t0 = time.time()
            cycles, mm, mult_util = run_config(inst_file, rows, cols, in_ch, out_ch)
            elapsed = time.time() - t0
            rows_out.append(BenchRow(
                label=f"{rows}x{cols} ic={in_ch} oc={out_ch}",
                cycles=cycles,
                mult_utilization=mult_util,
                correct=(mm == 0),
                elapsed_s=elapsed,
            ))

        out_path = Path(__file__).resolve().parent / "results.md"
        print_and_write_table("pointwise_conv_unified", rows_out, out_path)


if __name__ == "__main__":
    main()
