"""Benchmark depthwise_conv_universal across configs: cycles + correctness.

Runs each config through the emulator, compares output to a numpy IPU-math
reference for correctness, and reports cycles, theoretical minimum cycles
(num_chunks * channels * 9 cyc/ch — the absolute MULT-issue floor), and
end-to-end cyc/ch. Note depthwise's structural floor is 10 cyc/ch (see
depthwise_conv_universal.asm header for why), so realistic efficiency caps
at ~90%.

Usage::

    PYTHONPATH=... python -m ipu_apps.convolutions_universal.depthwise_conv_universal.benchmark.benchmark
"""

from __future__ import annotations

import math
import struct
import tempfile
import time
from pathlib import Path

import numpy as np

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.convolutions_universal.depthwise_conv_universal import (
    DepthwiseConvUniversalApp,
    OUTPUT_BASE_ADDR,
    OUTPUT_CHUNK_BYTES,
    FPB,
)


ASM_PATH = Path(__file__).resolve().parents[1] / "depthwise_conv_universal.asm"


# (rows, cols, channels) — covers spatial sizes, FPB=28 boundary cases
# (full block, partial last block), and small/large channel counts.
CONFIGS = [
    (16, 16, 1),       # tiny: single channel, smallest spatial
    (16, 16, 8),       # partial single block
    (16, 16, 28),      # exactly 1 full block
    (16, 16, 29),      # 1 full + 1-channel partial
    (32, 32, 16),      # multi-chunk, partial block
    (32, 32, 32),      # multi-chunk, 1 full + partial
    (32, 32, 56),      # exactly 2 full blocks
    (64, 64, 32),      # primary benchmark — large spatial
    (64, 64, 64),      # large spatial, 2 full + partial
    (32, 32, 128),     # many channels, 4 full + partial
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


def reference_depthwise(
    weights: np.ndarray, input_chw: np.ndarray, rows: int, cols: int
) -> bytes:
    """Numpy 3x3 depthwise conv with zero-padding, int32 wrapping accumulation.

    INT8 multiply produces int32; numpy int32 wraps on overflow, matching IPU.
    """
    channels = weights.shape[0]
    rows_per_chunk = 128 // cols
    num_chunks = (rows * cols) // 128

    inp32 = input_chw.astype(np.int32)
    padded = np.pad(inp32, ((0, 0), (1, 1), (1, 1)), mode="constant")
    w32 = weights.astype(np.int32)  # (channels, 3, 3)

    result = np.zeros((channels, rows, cols), dtype=np.int32)
    for dr in range(3):
        for dc in range(3):
            patch = padded[:, dr:dr + rows, dc:dc + cols]  # (channels, rows, cols)
            # Per-channel multiply (no in-channel reduction):
            result += w32[:, dr, dc][:, None, None] * patch

    output = bytearray(num_chunks * channels * 512)
    for ch in range(channels):
        for r in range(rows):
            chunk = r // rows_per_chunk
            local_row = r % rows_per_chunk
            for c in range(cols):
                elem = local_row * cols + c
                out_idx = (chunk * channels + ch) * 512 + elem * 4
                struct.pack_into("<i", output, out_idx, result[ch, r, c].item())
    return bytes(output)


def compare(actual: bytes, expected: bytes) -> int:
    mismatches = 0
    for i in range(0, len(expected), 4):
        a = struct.unpack_from("<i", actual, i)[0]
        e = struct.unpack_from("<i", expected, i)[0]
        if a != e:
            mismatches += 1
    return mismatches


def run_config(inst_file: Path, rows: int, cols: int, channels: int):
    rng = np.random.RandomState(42 + channels * 7 + rows + cols)
    weights = rng.randint(-32, 33, size=(channels, 3, 3), dtype=np.int8)
    input_chw = rng.randint(-32, 33, size=(channels, rows, cols), dtype=np.int8)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        input_file = tmp / "input.bin"
        input_file.write_bytes(pack_input_chunked(input_chw, rows, cols))

        kernel_file = tmp / "kernel.bin"
        # Raw layout: channel ch's 9 weights at byte offset ch*9 (row-major dr,dc).
        kernel_file.write_bytes(weights.reshape(channels, 9).tobytes())

        app = DepthwiseConvUniversalApp(
            inst_path=inst_file,
            input_path=input_file,
            kernel_path=kernel_file,
            output_path=None,
            dtype="INT8",
            rows=rows, cols=cols, channels=channels,
        )

        num_chunks = (rows * cols) // 128
        max_cyc = 2000 * num_chunks * math.ceil(channels / FPB) + 50_000
        state, cycles = app.run(max_cycles=max_cyc)

        total_bytes = num_chunks * channels * OUTPUT_CHUNK_BYTES
        actual = state.xmem.read_address(OUTPUT_BASE_ADDR, total_bytes)
        expected = reference_depthwise(weights, input_chw, rows, cols)

    mismatches = compare(actual, expected)
    return cycles, mismatches, num_chunks


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        inst_file = Path(tmp) / "depthwise_conv_universal.bin"
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
            # Theoretical min: 9 cyc/ch (one cycle per mult tap, 100% MULT
            # util). Depthwise's structural floor is 10 cyc/ch due to the
            # per-channel store cycle, so achievable efficiency caps near 90%.
            theory = 9 * channels_processed
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
