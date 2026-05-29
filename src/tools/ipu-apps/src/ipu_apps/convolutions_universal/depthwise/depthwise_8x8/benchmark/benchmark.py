"""Benchmark depthwise_8x8 across channel configs: cycles + correctness.

Fixed 8x8 spatial (64 positions). Paired-channel processing: channels 2k and
2k+1 share one accumulator (even ch in lanes 0-63, odd ch in lanes 64-127).
Output: 128-byte int8 per pair (AAQ-quantized, clamp [-128, 127]).

Reports cycles, theoretical minimum (9 * 64 * channels / 128 — MULT-issue
floor at 128 mults/cycle), and end-to-end efficiency.

Usage::

    PYTHONPATH=... python -m ipu_apps.convolutions_universal.depthwise.depthwise_8x8.benchmark.benchmark
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.convolutions_universal.depthwise.depthwise_8x8 import (
    Depthwise8x8App,
    OUTPUT_BASE_ADDR,
    OUTPUT_CHUNK_BYTES,
    ROWS,
    COLS,
    SPATIAL,
    _build_input_data,
    _build_kernel_data,
)


ASM_PATH = Path(__file__).resolve().parents[1] / "depthwise_8x8.asm"

# (channels,) — multiples of 8; covers small, partial-group, and large counts.
CONFIGS = [8, 16, 32, 64, 80, 128, 160, 256]


def pack_input(input_chw: np.ndarray, channels: int) -> bytes:
    raw = input_chw.reshape(channels, SPATIAL).view(np.uint8).tobytes()
    return _build_input_data(raw, channels)


def reference_depthwise(weights: np.ndarray, input_chw: np.ndarray) -> bytes:
    """Per-channel 3x3 depthwise conv, int32 acc, int8-clamped output.

    Output layout: num_pairs * 128 bytes, pairs interleaved (ch_even in
    bytes 0-63, ch_odd in 64-127 of each 128-byte record).
    """
    channels = weights.shape[0]
    num_pairs = channels // 2
    inp32 = input_chw.astype(np.int32)
    padded = np.pad(inp32, ((0, 0), (1, 1), (1, 1)), mode="constant")
    w32 = weights.astype(np.int32)
    result = np.zeros((channels, ROWS, COLS), dtype=np.int32)
    for dr in range(3):
        for dc in range(3):
            patch = padded[:, dr:dr + ROWS, dc:dc + COLS]
            result += w32[:, dr, dc][:, None, None] * patch
    clamped = np.clip(result, -128, 127).astype(np.int8).reshape(channels, SPATIAL)
    out = bytearray(num_pairs * OUTPUT_CHUNK_BYTES)
    for p in range(num_pairs):
        out[p * 128: p * 128 + 64] = clamped[2 * p].view(np.uint8).tobytes()
        out[p * 128 + 64: p * 128 + 128] = clamped[2 * p + 1].view(np.uint8).tobytes()
    return bytes(out)


def run_config(inst_file: Path, channels: int):
    rng = np.random.RandomState(42 + channels * 7)
    weights = rng.randint(-4, 5, size=(channels, 3, 3), dtype=np.int8)
    input_chw = rng.randint(-8, 9, size=(channels, ROWS, COLS), dtype=np.int8)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        kernel_raw = weights.reshape(channels, 9).view(np.uint8).tobytes()
        input_packed = pack_input(input_chw, channels)
        kernel_packed = _build_kernel_data(kernel_raw, channels)

        input_file = tmp / "input.bin"
        kernel_file = tmp / "kernel.bin"
        input_file.write_bytes(input_packed)
        kernel_file.write_bytes(kernel_packed)

        app = Depthwise8x8App(
            inst_path=inst_file,
            input_path=input_file,
            kernel_path=kernel_file,
            output_path=None,
            channels=channels,
        )
        num_pairs = channels // 2
        max_cyc = 2000 * channels + 50_000
        state, cycles = app.run(max_cycles=max_cyc)

        actual = state.xmem.read_address(OUTPUT_BASE_ADDR, num_pairs * OUTPUT_CHUNK_BYTES)
        expected = reference_depthwise(weights, input_chw)

    mismatches = sum(a != b for a, b in zip(actual, expected))
    return cycles, mismatches


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        inst_file = Path(tmp) / "depthwise_8x8.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(inst_file))

        header = (
            f"{'config':>12} {'cycles':>10} {'MAC/cyc':>8} "
            f"{'theory':>10} {'eff%':>6} {'corr':>5} {'time(s)':>8}"
        )
        print(header)
        print("-" * len(header))

        total_cycles = 0
        total_theory = 0
        all_correct = True

        for channels in CONFIGS:
            t0 = time.time()
            cycles, mismatches = run_config(inst_file, channels)
            elapsed = time.time() - t0

            # Theoretical MULT-issue floor: 9 * 64 * channels / 128
            theory = 9 * SPATIAL * channels // 128
            mac_per_cyc = (9 * SPATIAL * channels) / cycles
            efficiency = theory / cycles * 100

            ok = mismatches == 0
            all_correct = all_correct and ok

            label = f"ch={channels}"
            print(
                f"{label:>12} {cycles:>10} {mac_per_cyc:>8.2f} "
                f"{theory:>10} {efficiency:>5.1f}% "
                f"{'PASS' if ok else 'FAIL':>5} {elapsed:>8.2f}"
            )

            total_cycles += cycles
            total_theory += theory

        print("-" * len(header))
        print(
            f"{'TOTAL':>12} {total_cycles:>10} {'':>8} "
            f"{total_theory:>10} {total_theory/total_cycles*100:>5.1f}% "
            f"{'PASS' if all_correct else 'FAIL':>5}"
        )


if __name__ == "__main__":
    main()
