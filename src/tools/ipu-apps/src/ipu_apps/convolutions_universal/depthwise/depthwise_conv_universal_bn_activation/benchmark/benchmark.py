"""Benchmark depthwise_conv_universal_bn_activation: cycles + correctness.

Like the depthwise_conv_universal benchmark, but the reference is depthwise conv
+ per-channel folded bias + ReLU + INT8 clamp (matching the app's bias-seed ->
ACTIVATE relu -> AAQ pipeline).

The structural floor here is **10 cyc/ch** (1 bias-seed cycle + 9 weight taps),
so the theory column uses 10 cyc/ch and efficiency caps near 100% at that floor.

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
    OUTPUT_BASE_ADDR,
    OUTPUT_CHUNK_BYTES,
    FPB,
)


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
        actual = state.xmem.read_address(OUTPUT_BASE_ADDR, total_bytes)
        expected = reference_depthwise_bn_relu(weights, input_chw, bias, rows, cols)

    return cycles, compare(actual, expected), num_chunks


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        inst_file = Path(tmp) / "depthwise_conv_universal_bn_activation.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(inst_file))

        header = (
            f"{'config':>16} {'cycles':>10} {'cyc/ch':>8} "
            f"{'theory':>10} {'eff%':>6} {'corr':>5} {'time(s)':>8}"
        )
        print(header)
        print("-" * len(header))

        total_cycles = 0
        total_theory = 0
        all_correct = True

        for rows, cols, channels in CONFIGS:
            t0 = time.time()
            cycles, mismatches, num_chunks = run_config(
                inst_file, rows, cols, channels
            )
            elapsed = time.time() - t0

            channels_processed = num_chunks * channels
            cyc_per_ch = cycles / channels_processed
            # Structural floor: 10 cyc/ch (1 bias-seed + 9 weight taps).
            theory = 10 * channels_processed
            efficiency = theory / cycles * 100

            ok = mismatches == 0
            all_correct = all_correct and ok

            label = f"{rows}x{cols}x{channels}"
            print(
                f"{label:>16} {cycles:>10} {cyc_per_ch:>8.3f} "
                f"{theory:>10} {efficiency:>5.1f}% "
                f"{'PASS' if ok else 'FAIL':>5} {elapsed:>8.2f}"
            )

            total_cycles += cycles
            total_theory += theory

        print("-" * len(header))
        print(
            f"{'TOTAL':>16} {total_cycles:>10} {'':>8} "
            f"{total_theory:>10} {total_theory/total_cycles*100:>5.1f}% "
            f"{'PASS' if all_correct else 'FAIL':>5}"
        )


if __name__ == "__main__":
    main()
