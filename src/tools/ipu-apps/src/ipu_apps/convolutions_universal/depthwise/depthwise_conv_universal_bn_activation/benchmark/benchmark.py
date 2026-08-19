"""Benchmark depthwise_conv_universal_bn_activation: cycles + real MULT util.

Like the depthwise_conv_universal benchmark, but the reference is depthwise conv
+ per-channel folded bias + ReLU + INT8 clamp (matching the app's bias-seed ->
ACTIVATE relu -> AAQ pipeline).

Reports cycles and MULT-slot utilization read directly from ``state.stats``
(the emulator's live per-cycle occupancy counter).

Usage::

    PYTHONPATH=... python -m ipu_apps.convolutions_universal.depthwise.depthwise_conv_universal_bn_activation.benchmark.benchmark
"""

from __future__ import annotations

import math
import tempfile
import time
from pathlib import Path

import numpy as np

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.convolutions_universal.depthwise.depthwise_conv_universal_bn_activation import (
    DepthwiseConvUniversalBnActivationApp,
    OUTPUT_CHUNK_BYTES,
    FPB,
)
from ipu_apps.convolutions_universal.benchmarking import BenchRow, print_and_write_table


ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "depthwise_conv_universal_bn_activation.asm"
)


# (rows, cols, channels) — spatial sizes + FPB=25 boundary cases.
CONFIGS = [
    (16, 16, 8),       # partial single block
    (16, 16, 25),      # exactly 1 full FPB=25 block
    (16, 16, 26),      # 1 full + 1-channel partial
    (32, 32, 16),      # multi-chunk, partial block
    (32, 32, 32),      # multi-chunk, 1 full + partial
    (32, 32, 50),      # exactly 2 full blocks
    (64, 64, 32),      # primary benchmark — large spatial
    (64, 64, 64),      # large spatial, multiple blocks
    (32, 32, 96),      # many channels
    (16, 16, 40),      # two blocks, small spatial
]


def pack_input_chunked(input_chw: np.ndarray, rows: int, cols: int) -> bytes:
    channels = input_chw.shape[0]
    rows_per_chunk = 128 // cols
    num_chunks = (rows * cols) // 128
    packed = bytearray(num_chunks * channels * 128)
    for ch in range(channels):
        for r in range(rows):
            for c in range(cols):
                chunk = r // rows_per_chunk
                local_row = r % rows_per_chunk
                offset = (chunk * channels + ch) * 128 + local_row * cols + c
                packed[offset] = np.uint8(input_chw[ch, r, c]).item()
    return bytes(packed)


def reference_depthwise_bn_relu(
    weights: np.ndarray, input_chw: np.ndarray, bias: np.ndarray,
    rows: int, cols: int,
) -> bytes:
    """3x3 depthwise conv (zero-pad) + per-channel bias + ReLU + int8 clamp."""
    channels = weights.shape[0]
    rows_per_chunk = 128 // cols
    num_chunks = (rows * cols) // 128

    inp32 = input_chw.astype(np.int32)
    padded = np.pad(inp32, ((0, 0), (1, 1), (1, 1)), mode="constant")
    w32 = weights.astype(np.int32)  # (channels, 3, 3)

    result = np.zeros((channels, rows, cols), dtype=np.int32)
    for dr in range(3):
        for dc in range(3):
            patch = padded[:, dr:dr + rows, dc:dc + cols]
            result += w32[:, dr, dc][:, None, None] * patch

    result += bias.astype(np.int32)[:, None, None]   # folded bias
    result = np.maximum(result, 0)                    # ReLU
    clamped = np.clip(result, -128, 127).astype(np.int8)

    output = bytearray(num_chunks * channels * 128)
    for ch in range(channels):
        for r in range(rows):
            chunk = r // rows_per_chunk
            local_row = r % rows_per_chunk
            for c in range(cols):
                elem = local_row * cols + c
                out_idx = (chunk * channels + ch) * 128 + elem
                output[out_idx] = np.uint8(clamped[ch, r, c]).item()
    return bytes(output)


def compare(actual: bytes, expected: bytes) -> int:
    return sum(1 for i in range(len(expected)) if actual[i] != expected[i])


def run_config(inst_file: Path, rows: int, cols: int, channels: int):
    rng = np.random.RandomState(42 + channels * 7 + rows + cols)
    weights = rng.randint(-4, 5, size=(channels, 3, 3), dtype=np.int8)
    input_chw = rng.randint(-8, 9, size=(channels, rows, cols), dtype=np.int8)
    bias = rng.randint(-80, 81, size=channels).astype(np.int8)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        input_file = tmp / "input.bin"
        input_file.write_bytes(pack_input_chunked(input_chw, rows, cols))

        kernel_file = tmp / "kernel.bin"
        kernel_file.write_bytes(weights.reshape(channels, 9).tobytes())

        app = DepthwiseConvUniversalBnActivationApp(
            inst_path=inst_file,
            input_path=input_file,
            kernel_path=kernel_file,
            bias=bias,
            output_path=None,
            dtype="INT8",
            rows=rows, cols=cols, channels=channels,
        )

        num_chunks = (rows * cols) // 128
        max_cyc = 2000 * num_chunks * channels * math.ceil(channels / FPB) + 50_000
        state, cycles = app.run(max_cycles=max_cyc)

        total_bytes = num_chunks * channels * OUTPUT_CHUNK_BYTES
        actual = state.xmem.read_address(app.output_base_addr, total_bytes)
        expected = reference_depthwise_bn_relu(weights, input_chw, bias, rows, cols)

    return cycles, compare(actual, expected), state.stats.mult_utilization


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        inst_file = Path(tmp) / "depthwise_conv_universal_bn_activation.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(inst_file))

        rows_out = []
        for rows, cols, channels in CONFIGS:
            t0 = time.time()
            cycles, mismatches, mult_util = run_config(
                inst_file, rows, cols, channels
            )
            elapsed = time.time() - t0
            rows_out.append(BenchRow(
                label=f"{rows}x{cols}x{channels}",
                cycles=cycles,
                mult_utilization=mult_util,
                correct=(mismatches == 0),
                elapsed_s=elapsed,
            ))

        out_path = Path(__file__).resolve().parent / "results.md"
        print_and_write_table("depthwise_conv_universal_bn_activation", rows_out, out_path)


if __name__ == "__main__":
    main()
