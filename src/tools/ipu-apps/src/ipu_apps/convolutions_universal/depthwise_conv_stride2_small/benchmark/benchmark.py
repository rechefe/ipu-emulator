"""Benchmark depthwise_conv_stride2_small across configs: cycles + correctness.

Handles cols=32 or cols=64 (Jinja-templated binary — one per cols value,
assembled fresh into a temp directory). Output is INT8 (clamped to [-128, 127]).

Reports cycles, theoretical minimum (9 * out_rows * out_cols * channels / 128
— MULT-issue floor at 128 mults/cycle), and efficiency.

Usage::

    PYTHONPATH=... python -m ipu_apps.convolutions_universal.depthwise_conv_stride2_small.benchmark.benchmark
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.convolutions_universal.depthwise_conv_stride2_small import (
    DepthwiseConvStride2SmallApp,
    OUTPUT_BASE_ADDR,
)


ASM_TEMPLATE_PATH = (
    Path(__file__).resolve().parents[1] / "depthwise_conv_stride2_small.asm"
)

# (rows, cols, channels) — cols in {32, 64}; channels multiples of 8;
# num_groups = (rows*cols//128)//4 must be >= 2.
CONFIGS = [
    (64,  64,   8),
    (64,  64,  32),
    (128, 64,  64),
    (128, 64, 256),
    (64,  32,  16),
    (64,  32,  64),
    (128, 32, 384),
    (128, 32, 512),
]


def pack_input_multichannel(
    input_chw: np.ndarray, rows: int, cols: int, channels: int
) -> bytes:
    """Pack CHW input into multi-channel chunked layout (128B/ch/row-group)."""
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


def reference_depthwise_stride2(
    weights: np.ndarray, input_chw: np.ndarray, rows: int, cols: int
) -> np.ndarray:
    """Stride-2 depthwise 3x3 conv + int8 clamp. Returns (channels, rows//2, cols//2) int8."""
    channels = weights.shape[0]
    inp32 = input_chw.astype(np.int32)
    padded = np.pad(inp32, ((0, 0), (1, 1), (1, 1)), mode="constant")
    w32 = weights.astype(np.int32)

    result = np.zeros((channels, rows, cols), dtype=np.int32)
    for dr in range(3):
        for dc in range(3):
            patch = padded[:, dr:dr + rows, dc:dc + cols]
            result += w32[:, dr, dc][:, None, None] * patch

    strided = result[:, 1::2, ::2]  # (channels, rows//2, cols//2)
    return strided.clip(-128, 127).astype(np.int8)


def read_output(
    state, rows: int, cols: int, channels: int
) -> np.ndarray:
    """Read INT8 multi-channel output. Returns (channels, rows//2, cols//2) int8."""
    out_cols = cols // 2
    out_rows = rows // 2
    rows_per_out_chunk = 128 // out_cols
    num_chunks = (rows * cols) // 128
    num_groups = num_chunks // 4  # output groups

    raw = state.xmem.read_address(OUTPUT_BASE_ADDR, num_groups * channels * 128)
    vals = np.frombuffer(raw, dtype=np.uint8).reshape(num_groups, channels, 128)
    result = np.empty((channels, out_rows, out_cols), dtype=np.int8)
    for rg in range(num_groups):
        for ch in range(channels):
            block = vals[rg, ch, :rows_per_out_chunk * out_cols].view(np.int8)
            result[ch, rg * rows_per_out_chunk:(rg + 1) * rows_per_out_chunk, :] = (
                block.reshape(rows_per_out_chunk, out_cols)
            )
    return result


def run_config(
    inst_files: dict[int, Path], rows: int, cols: int, channels: int
):
    rng = np.random.RandomState(42 + channels * 7 + rows + cols)
    weights = rng.randint(-32, 33, size=(channels, 3, 3), dtype=np.int8)
    input_chw = rng.randint(-32, 33, size=(channels, rows, cols), dtype=np.int8)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        input_packed = pack_input_multichannel(input_chw, rows, cols, channels)
        kernel_bytes = weights.reshape(channels, 9).view(np.uint8).tobytes()

        input_file = tmp / "input.bin"
        kernel_file = tmp / "kernel.bin"
        input_file.write_bytes(input_packed)
        kernel_file.write_bytes(kernel_bytes)

        app = DepthwiseConvStride2SmallApp(
            inst_path=inst_files[cols],
            input_path=input_file,
            kernel_path=kernel_file,
            output_path=None,
            dtype="INT8",
            rows=rows, cols=cols, channels=channels,
        )
        max_cyc = 2000 * rows * channels + 100_000
        state, cycles = app.run(max_cycles=max_cyc)

        actual = read_output(state, rows, cols, channels)
        expected = reference_depthwise_stride2(weights, input_chw, rows, cols)

    mismatches = int(np.sum(actual != expected))
    return cycles, mismatches


def main() -> None:
    template_src = ASM_TEMPLATE_PATH.read_text()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        # Assemble one binary per unique cols value
        inst_files: dict[int, Path] = {}
        for cols in sorted({c for _, c, _ in CONFIGS}):
            rendered = f"{{% set cols = {cols} %}}\n" + template_src
            inst_file = tmp_dir / f"depthwise_conv_stride2_small_cols{cols}.bin"
            print(f"  Assembling cols={cols} ...", flush=True)
            assemble_to_bin_file(rendered, str(inst_file))
            inst_files[cols] = inst_file

        header = (
            f"{'config':>22} {'cycles':>10} {'MAC/cyc':>8} "
            f"{'theory':>10} {'eff%':>6} {'corr':>5} {'time(s)':>8}"
        )
        print(header)
        print("-" * len(header))

        total_cycles = 0
        total_theory = 0
        all_correct = True

        for rows, cols, channels in CONFIGS:
            t0 = time.time()
            cycles, mismatches = run_config(inst_files, rows, cols, channels)
            elapsed = time.time() - t0

            out_rows = rows // 2
            out_cols = cols // 2
            theory = 9 * out_rows * out_cols * channels // 128
            mac_per_cyc = (9 * out_rows * out_cols * channels) / cycles
            efficiency = theory / cycles * 100

            ok = mismatches == 0
            all_correct = all_correct and ok

            label = f"r={rows} c={cols} ch={channels}"
            print(
                f"{label:>22} {cycles:>10} {mac_per_cyc:>8.2f} "
                f"{theory:>10} {efficiency:>5.1f}% "
                f"{'PASS' if ok else 'FAIL':>5} {elapsed:>8.2f}"
            )

            total_cycles += cycles
            total_theory += theory

        print("-" * len(header))
        print(
            f"{'TOTAL':>22} {total_cycles:>10} {'':>8} "
            f"{total_theory:>10} {total_theory/total_cycles*100:>5.1f}% "
            f"{'PASS' if all_correct else 'FAIL':>5}"
        )


if __name__ == "__main__":
    main()
