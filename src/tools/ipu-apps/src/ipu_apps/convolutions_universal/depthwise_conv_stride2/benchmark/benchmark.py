"""Benchmark depthwise_conv_stride2 across configs: cycles + correctness.

Fixed cols=128 (one spatial row per 128-byte chunk). Stride-2 reduces both
spatial dimensions by half; output is INT8 (quantized/clamped to [-128, 127]).

Reports cycles, theoretical minimum (9 * out_rows * 64 * channels / 128 —
MULT-issue floor at 128 mults/cycle), and efficiency.

Usage::

    PYTHONPATH=... python -m ipu_apps.convolutions_universal.depthwise_conv_stride2.benchmark.benchmark
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.convolutions_universal.depthwise_conv_stride2 import (
    DepthwiseConvStride2App,
)


ASM_PATH = Path(__file__).resolve().parents[1] / "depthwise_conv_stride2.asm"

COLS = 128  # fixed for this app

# rows must be 128 (ASM hardcodes 32 output chunks). For variable-row configs
# use depthwise_conv_stride2_small instead.
CONFIGS = [
    (128,  64),
    (128, 128),
    (128, 256),
    (128, 384),
]


def pack_input_multichannel(
    input_chw: np.ndarray, rows: int, channels: int
) -> bytes:
    """Pack CHW input into multi-channel chunked layout (128B/ch/row-group)."""
    rows_per_chunk = COLS  # 128 // 128 = 1
    row_groups = (rows * COLS) // 128  # = rows
    packed = bytearray(row_groups * channels * 128)
    for rg in range(row_groups):
        for ch in range(channels):
            dst = (rg * channels + ch) * 128
            packed[dst:dst + COLS] = input_chw[ch, rg, :].view(np.uint8).tobytes()
    return bytes(packed)


def reference_depthwise_stride2(
    weights: np.ndarray, input_chw: np.ndarray, rows: int
) -> np.ndarray:
    """Stride-2 depthwise 3x3 conv + int8 clamp. Returns (channels, rows//2, 64) int8."""
    channels = weights.shape[0]
    out_rows = rows // 2
    out_cols = COLS // 2  # 64

    inp32 = input_chw.astype(np.int32)
    padded = np.pad(inp32, ((0, 0), (1, 1), (1, 1)), mode="constant")
    w32 = weights.astype(np.int32)

    result = np.zeros((channels, rows, COLS), dtype=np.int32)
    for dr in range(3):
        for dc in range(3):
            patch = padded[:, dr:dr + rows, dc:dc + COLS]
            result += w32[:, dr, dc][:, None, None] * patch

    # Stride-2 decimation
    strided = result[:, 1::2, ::2]  # (channels, out_rows, out_cols)
    return strided.clip(-128, 127).astype(np.int8)


def read_output(
    state, output_base: int, rows: int, channels: int
) -> np.ndarray:
    """Read INT8 multi-channel output. Returns (channels, out_rows, out_cols) int8."""
    out_rows = rows // 2
    out_cols = COLS // 2  # 64
    rows_per_out_chunk = 128 // out_cols  # = 2
    num_out_chunks = out_rows // rows_per_out_chunk

    raw = state.xmem.read_address(output_base, num_out_chunks * channels * 128)
    vals = np.frombuffer(raw, dtype=np.uint8).reshape(num_out_chunks, channels, 128)
    result = np.empty((channels, out_rows, out_cols), dtype=np.int8)
    for rg in range(num_out_chunks):
        for ch in range(channels):
            block = vals[rg, ch, :rows_per_out_chunk * out_cols].view(np.int8)
            result[ch, rg * rows_per_out_chunk:(rg + 1) * rows_per_out_chunk, :] = (
                block.reshape(rows_per_out_chunk, out_cols)
            )
    return result


def run_config(inst_file: Path, rows: int, channels: int):
    rng = np.random.RandomState(42 + channels * 7 + rows)
    weights = rng.randint(-32, 33, size=(channels, 3, 3), dtype=np.int8)
    input_chw = rng.randint(-32, 33, size=(channels, rows, COLS), dtype=np.int8)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        input_packed = pack_input_multichannel(input_chw, rows, channels)
        kernel_bytes = weights.reshape(channels, 9).view(np.uint8).tobytes()

        input_file = tmp / "input.bin"
        kernel_file = tmp / "kernel.bin"
        input_file.write_bytes(input_packed)
        kernel_file.write_bytes(kernel_bytes)

        app = DepthwiseConvStride2App(
            inst_path=inst_file,
            input_path=input_file,
            kernel_path=kernel_file,
            output_path=None,
            dtype="INT8",
            rows=rows, cols=COLS, channels=channels,
        )
        max_cyc = 2000 * rows * channels + 50_000
        state, cycles = app.run(max_cycles=max_cyc)

        actual = read_output(state, app.output_base, rows, channels)
        expected = reference_depthwise_stride2(weights, input_chw, rows)

    mismatches = int(np.sum(actual != expected))
    return cycles, mismatches


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        inst_file = Path(tmp) / "depthwise_conv_stride2.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(inst_file))

        header = (
            f"{'config':>18} {'cycles':>10} {'MAC/cyc':>8} "
            f"{'theory':>10} {'eff%':>6} {'corr':>5} {'time(s)':>8}"
        )
        print(header)
        print("-" * len(header))

        total_cycles = 0
        total_theory = 0
        all_correct = True

        for rows, channels in CONFIGS:
            t0 = time.time()
            cycles, mismatches = run_config(inst_file, rows, channels)
            elapsed = time.time() - t0

            out_rows = rows // 2
            out_cols = COLS // 2  # 64
            # Theory: 9 * out_rows * out_cols * channels / 128
            theory = 9 * out_rows * out_cols * channels // 128
            mac_per_cyc = (9 * out_rows * out_cols * channels) / cycles
            efficiency = theory / cycles * 100

            ok = mismatches == 0
            all_correct = all_correct and ok

            label = f"r={rows} ch={channels}"
            print(
                f"{label:>18} {cycles:>10} {mac_per_cyc:>8.2f} "
                f"{theory:>10} {efficiency:>5.1f}% "
                f"{'PASS' if ok else 'FAIL':>5} {elapsed:>8.2f}"
            )

            total_cycles += cycles
            total_theory += theory

        print("-" * len(header))
        print(
            f"{'TOTAL':>18} {total_cycles:>10} {'':>8} "
            f"{total_theory:>10} {total_theory/total_cycles*100:>5.1f}% "
            f"{'PASS' if all_correct else 'FAIL':>5}"
        )


if __name__ == "__main__":
    main()
