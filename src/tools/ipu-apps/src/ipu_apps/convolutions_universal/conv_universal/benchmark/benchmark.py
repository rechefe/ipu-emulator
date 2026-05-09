"""Benchmark conv_universal across 10 configs: cycles + correctness.

Runs each config through the emulator, compares output to a numpy IPU-math
reference for correctness, and reports cycles, theoretical minimum cycles
(num_chunks * out_ch * ceil(in_ch/14) * 14 * 9 taps = mult-issued cycles),
and end-to-end cyc/ch.

Usage::

    PYTHONPATH=... python -m ipu_apps.convolutions_universal.conv_universal.benchmark.benchmark
"""

from __future__ import annotations

import math
import struct
import tempfile
import time
from pathlib import Path

import numpy as np

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.convolutions_universal.conv_universal import (
    ConvUniversalApp,
    OUTPUT_BASE_ADDR,
    OUTPUT_CHUNK_BYTES,
)


ASM_PATH = Path(__file__).resolve().parents[1] / "conv_universal.asm"


# (rows, cols, in_ch, out_ch) — 10 configs covering shapes, channel counts,
# partial vs full last blocks, and small/large spatial sizes.
CONFIGS = [
    (16, 16, 14, 4),     # exactly 1 full block, smallest spatial
    (16, 16, 16, 4),     # partial last block (16 % 14 = 2)
    (16, 16, 28, 4),     # exactly 2 full blocks
    (32, 32, 14, 8),     # 1 full block, multi-chunk
    (32, 32, 32, 16),    # partial last block, multi-filter
    (32, 32, 16, 32),    # partial last block, more filters than channels
    (64, 64, 32, 32),    # primary benchmark — large
    (64, 64, 14, 14),    # 1 full block, larger spatial
    (32, 32, 64, 8),     # large in_ch, multiple full+partial blocks
    (16, 16, 42, 4),     # exactly 3 full blocks
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


def reference_conv(
    weights: np.ndarray, input_chw: np.ndarray, rows: int, cols: int
) -> bytes:
    """Plain numpy 3x3 conv with zero-padding, int32 accumulation (wraps at 32 bits).

    INT8 multiply produces int32; accumulation is int32 wrapping — matches IPU
    exactly for INT8 dtype (ipu_add wraps at 2^32 with two's complement).
    """
    out_ch = weights.shape[0]
    rows_per_chunk = 128 // cols
    num_chunks = (rows * cols) // 128

    # Zero-pad input: (in_ch, rows+2, cols+2)
    inp32 = input_chw.astype(np.int32)
    padded = np.pad(inp32, ((0, 0), (1, 1), (1, 1)), mode="constant")

    # Build (out_ch, rows, cols) accumulator via einsum over 3x3 kernel and in_ch.
    # Use int32 throughout; numpy int32 wraps on overflow, matching IPU behaviour.
    w32 = weights.astype(np.int32)  # (out_ch, in_ch, 3, 3)

    # Collect all 9 shifted views and accumulate.
    result = np.zeros((out_ch, rows, cols), dtype=np.int32)
    for dr in range(3):
        for dc in range(3):
            patch = padded[:, dr:dr + rows, dc:dc + cols]  # (in_ch, rows, cols)
            # w32[:, :, dr, dc] is (out_ch, in_ch); patch is (in_ch, rows, cols)
            result += np.tensordot(w32[:, :, dr, dc], patch, axes=([1], [0]))

    # Pack into chunked output layout: (chunk * out_ch + f) * 512 + elem * 4
    output = bytearray(num_chunks * out_ch * 512)
    for f in range(out_ch):
        for r in range(rows):
            chunk = r // rows_per_chunk
            local_row = r % rows_per_chunk
            for c in range(cols):
                elem = local_row * cols + c
                out_idx = (chunk * out_ch + f) * 512 + elem * 4
                struct.pack_into("<i", output, out_idx, result[f, r, c].item())
    return bytes(output)


def compare(actual: bytes, expected: bytes, out_ch: int, cols: int) -> int:
    rows_per_chunk = 128 // cols
    mismatches = 0
    for i in range(0, len(expected), 4):
        a = struct.unpack_from("<i", actual, i)[0]
        e = struct.unpack_from("<i", expected, i)[0]
        if a != e:
            mismatches += 1
    return mismatches


def run_config(inst_file: Path, rows: int, cols: int, in_ch: int, out_ch: int):
    rng = np.random.RandomState(42 + in_ch * 7 + out_ch * 13 + rows + cols)
    weights = rng.randint(-32, 33, size=(out_ch, in_ch, 3, 3), dtype=np.int8)
    input_chw = rng.randint(-32, 33, size=(in_ch, rows, cols), dtype=np.int8)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        input_file = tmp / "input.bin"
        input_file.write_bytes(pack_input_chunked(input_chw, rows, cols))

        app = ConvUniversalApp(
            inst_path=inst_file,
            input_path=input_file,
            kernel=weights,
            output_path=None,
            dtype="INT8",
            rows=rows, cols=cols,
            in_channels=in_ch, out_channels=out_ch,
        )

        num_chunks = (rows * cols) // 128
        max_cyc = 2000 * num_chunks * out_ch * math.ceil(in_ch / 14) + 50_000
        state, cycles = app.run(max_cycles=max_cyc)

        total_bytes = num_chunks * out_ch * OUTPUT_CHUNK_BYTES
        actual = state.xmem.read_address(OUTPUT_BASE_ADDR, total_bytes)
        expected = reference_conv(weights, input_chw, rows, cols)

    mismatches = compare(actual, expected, out_ch, cols)
    return cycles, mismatches, num_chunks


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        inst_file = Path(tmp) / "conv_universal.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(inst_file))

        header = (
            f"{'config':>20} {'cycles':>10} {'cyc/ch':>8} "
            f"{'theory':>10} {'eff%':>6} {'corr':>5} {'time(s)':>8}"
        )
        print(header)
        print("-" * len(header))

        total_cycles = 0
        total_theory = 0
        all_correct = True

        for rows, cols, in_ch, out_ch in CONFIGS:
            t0 = time.time()
            cycles, mismatches, num_chunks = run_config(
                inst_file, rows, cols, in_ch, out_ch
            )
            elapsed = time.time() - t0

            blocks_per_filter = math.ceil(in_ch / 14)
            channels_processed = num_chunks * out_ch * in_ch
            cyc_per_ch = cycles / channels_processed
            # Theoretical min: 9 cyc/ch (mn body, 100% MULT util) — but
            # boundary chunks (g0, gN) and outer-loop overhead add more.
            theory = 9 * channels_processed
            efficiency = theory / cycles * 100

            ok = mismatches == 0
            all_correct = all_correct and ok

            label = f"{rows}x{cols}x{in_ch}x{out_ch}"
            print(
                f"{label:>20} {cycles:>10} {cyc_per_ch:>8.3f} "
                f"{theory:>10} {efficiency:>5.1f}% "
                f"{'PASS' if ok else 'FAIL':>5} {elapsed:>8.2f}"
            )

            total_cycles += cycles
            total_theory += theory

        print("-" * len(header))
        print(
            f"{'TOTAL':>20} {total_cycles:>10} {'':>8} "
            f"{total_theory:>10} {total_theory/total_cycles*100:>5.1f}% "
            f"{'PASS' if all_correct else 'FAIL':>5}"
        )


if __name__ == "__main__":
    main()
