"""Benchmark conv_universal_bn_activation across configs: cycles + real MULT util.

Like the conv_universal benchmark, but the reference is conv + per-filter folded
bias + ReLU + INT8 clamp (matching the app's ACTIVATE relu -> AAQ pipeline).

Runs each config through the emulator, compares output to a numpy IPU-math
reference, and reports cycles and MULT-slot utilization read directly from
``state.stats`` (the emulator's live per-cycle occupancy counter).

Usage::

    PYTHONPATH=... python -m ipu_apps.convolutions_universal.conv.conv_universal_bn_activation.benchmark.benchmark
"""

from __future__ import annotations

import math
import tempfile
import time
from pathlib import Path

import numpy as np

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.convolutions_universal.conv.conv_universal_bn_activation import (
    ConvUniversalBnActivationApp,
    OUTPUT_BASE_ADDR,
    OUTPUT_CHUNK_BYTES,
    FPB,
)
from ipu_apps.convolutions_universal.benchmarking import BenchRow, print_and_write_table


ASM_PATH = Path(__file__).resolve().parents[1] / "conv_universal_bn_activation.asm"


# (rows, cols, in_ch, out_ch) — covers shapes, channel counts, partial vs full
# last super-blocks (FPB=28), and small/large spatial sizes.
CONFIGS = [
    (16, 16, 14, 4),     # half a super-block, smallest spatial
    (16, 16, 16, 4),     # partial last block
    (16, 16, 28, 4),     # exactly one full super-block (bias byte + 28*9 = 253)
    (32, 32, 14, 8),     # multi-chunk
    (32, 32, 32, 16),    # two super-blocks, multi-filter
    (32, 32, 16, 32),    # more filters than channels
    (64, 64, 32, 32),    # primary benchmark — large
    (64, 64, 28, 14),    # one full super-block, larger spatial
    (32, 32, 64, 8),     # large in_ch, multiple super-blocks
    (16, 16, 40, 4),     # two super-blocks (bias accumulated once)
]


def pack_input_chunked(input_chw: np.ndarray, rows: int, cols: int) -> bytes:
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


def reference_conv_bn_relu(
    weights: np.ndarray, input_chw: np.ndarray, bias: np.ndarray, rows: int, cols: int
) -> bytes:
    """3x3 conv (zero-pad) + per-filter bias + ReLU + int8 clamp.

    int32 accumulation; bias seeds the accumulator; ReLU = max(0, .); output
    clamped to [-128, 127] (== [0, 127] after ReLU).
    """
    out_ch = weights.shape[0]
    rows_per_chunk = 128 // cols
    num_chunks = (rows * cols) // 128

    inp32 = input_chw.astype(np.int32)
    padded = np.pad(inp32, ((0, 0), (1, 1), (1, 1)), mode="constant")
    w32 = weights.astype(np.int32)

    result = np.zeros((out_ch, rows, cols), dtype=np.int32)
    for dr in range(3):
        for dc in range(3):
            patch = padded[:, dr:dr + rows, dc:dc + cols]
            result += np.tensordot(w32[:, :, dr, dc], patch, axes=([1], [0]))

    result += bias.astype(np.int32)[:, None, None]   # folded bias
    result = np.maximum(result, 0)                    # ReLU
    clamped = np.clip(result, -128, 127).astype(np.int8)

    output = bytearray(num_chunks * out_ch * 128)
    for f in range(out_ch):
        for r in range(rows):
            chunk = r // rows_per_chunk
            local_row = r % rows_per_chunk
            for c in range(cols):
                elem = local_row * cols + c
                out_idx = (chunk * out_ch + f) * 128 + elem
                output[out_idx] = np.uint8(clamped[f, r, c]).item()
    return bytes(output)


def compare(actual: bytes, expected: bytes) -> int:
    return sum(1 for i in range(len(expected)) if actual[i] != expected[i])


def run_config(inst_file: Path, rows: int, cols: int, in_ch: int, out_ch: int):
    rng = np.random.RandomState(42 + in_ch * 7 + out_ch * 13 + rows + cols)
    weights = rng.randint(-4, 5, size=(out_ch, in_ch, 3, 3), dtype=np.int8)
    input_chw = rng.randint(-8, 9, size=(in_ch, rows, cols), dtype=np.int8)
    bias = rng.randint(-80, 81, size=out_ch).astype(np.int8)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        input_file = tmp / "input.bin"
        input_file.write_bytes(pack_input_chunked(input_chw, rows, cols))

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
        max_cyc = 2000 * num_chunks * out_ch * math.ceil(in_ch / FPB) + 50_000
        state, cycles = app.run(max_cycles=max_cyc)

        total_bytes = num_chunks * out_ch * OUTPUT_CHUNK_BYTES
        actual = state.xmem.read_address(OUTPUT_BASE_ADDR, total_bytes)
        expected = reference_conv_bn_relu(weights, input_chw, bias, rows, cols)

    return cycles, compare(actual, expected), state.stats.mult_utilization


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        inst_file = Path(tmp) / "conv_universal_bn_activation.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(inst_file))

        rows_out = []
        for rows, cols, in_ch, out_ch in CONFIGS:
            t0 = time.time()
            cycles, mismatches, mult_util = run_config(
                inst_file, rows, cols, in_ch, out_ch
            )
            elapsed = time.time() - t0
            rows_out.append(BenchRow(
                label=f"{rows}x{cols}x{in_ch}x{out_ch}",
                cycles=cycles,
                mult_utilization=mult_util,
                correct=(mismatches == 0),
                elapsed_s=elapsed,
            ))

        out_path = Path(__file__).resolve().parent / "results.md"
        print_and_write_table("conv_universal_bn_activation", rows_out, out_path)


if __name__ == "__main__":
    main()
