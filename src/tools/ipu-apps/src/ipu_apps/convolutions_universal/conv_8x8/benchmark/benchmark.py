"""Benchmark conv_8x8 across channel configs: cycles + correctness.

Fixed 8x8 spatial (64 positions). Paired-filter output: two output filters
share one accumulator (even OC in lanes 0-63, odd OC in lanes 64-127).

Reports cycles, theoretical minimum (9 * 64 * in_ch * out_ch / 128 — MULT-issue
floor at 128 mults/cycle), and end-to-end MAC/cycle efficiency.

Usage::

    PYTHONPATH=... python -m ipu_apps.convolutions_universal.conv_8x8.benchmark.benchmark
"""

from __future__ import annotations

import struct
import tempfile
import time
from pathlib import Path

import numpy as np

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.convolutions_universal.conv_8x8 import (
    Conv8x8App,
    OUTPUT_BASE_ADDR,
    ACC_CHUNK_BYTES,
    ROWS,
    COLS,
    SPATIAL,
    _build_input_data,
    _build_kernel_data,
)


ASM_PATH = Path(__file__).resolve().parents[1] / "conv_8x8.asm"

# (in_ch, out_ch) — both even; covers small/medium/large channel counts,
# partial last block (not multiple of 8), and multi-block configs.
CONFIGS = [
    (8,   8),
    (16,  8),
    (32,  16),
    (48,  32),
    (64,  64),
    (128, 64),
    (160, 160),
    (320, 160),
]


def pack_input(input_chw: np.ndarray, in_ch: int) -> bytes:
    """Pack CHW input into paired-chunk format (ch_even 0-63, ch_odd 64-127)."""
    raw = input_chw.reshape(in_ch, SPATIAL).view(np.uint8).tobytes()
    return _build_input_data(raw, in_ch)


def reference_conv(
    weights: np.ndarray, input_chw: np.ndarray
) -> np.ndarray:
    """3x3 zero-padded conv, int32 accumulation wrapping. Returns (out_ch, ROWS, COLS)."""
    out_ch = weights.shape[0]
    in_ch = weights.shape[1]
    inp32 = input_chw.astype(np.int32)
    padded = np.pad(inp32, ((0, 0), (1, 1), (1, 1)), mode="constant")
    w32 = weights.astype(np.int32)
    result = np.zeros((out_ch, ROWS, COLS), dtype=np.int32)
    for dr in range(3):
        for dc in range(3):
            patch = padded[:, dr:dr + ROWS, dc:dc + COLS]
            result += np.tensordot(w32[:, :, dr, dc], patch, axes=([1], [0]))
    return result


def read_output(state, out_ch: int) -> np.ndarray:
    """Read paired-output ACC from emulator state. Returns (out_ch, SPATIAL) int32."""
    oc_pairs = out_ch // 2
    raw = state.xmem.read_address(OUTPUT_BASE_ADDR, oc_pairs * ACC_CHUNK_BYTES)
    vals = np.frombuffer(raw, dtype="<i4")  # (oc_pairs * 128,)
    vals = vals.reshape(oc_pairs, 128)
    result = np.empty((out_ch, SPATIAL), dtype=np.int32)
    for p in range(oc_pairs):
        result[2 * p]     = vals[p, :SPATIAL]
        result[2 * p + 1] = vals[p, SPATIAL:2 * SPATIAL]
    return result


def compare(actual: np.ndarray, expected: np.ndarray) -> int:
    return int(np.sum(actual != expected))


def run_config(inst_file: Path, in_ch: int, out_ch: int):
    rng = np.random.RandomState(42 + in_ch * 7 + out_ch * 13)
    weights = rng.randint(-32, 33, size=(out_ch, in_ch, 3, 3), dtype=np.int8)
    input_chw = rng.randint(-32, 33, size=(in_ch, ROWS, COLS), dtype=np.int8)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        kernel_raw = weights.view(np.uint8).tobytes()
        input_packed = pack_input(input_chw, in_ch)
        kernel_packed = _build_kernel_data(kernel_raw, in_ch, out_ch)

        input_file = tmp / "input.bin"
        kernel_file = tmp / "kernel.bin"
        input_file.write_bytes(input_packed)
        kernel_file.write_bytes(kernel_packed)

        app = Conv8x8App(
            inst_path=inst_file,
            input_path=input_file,
            kernel_path=kernel_file,
            output_path=None,
            in_channels=in_ch,
            out_channels=out_ch,
        )
        max_cyc = 2000 * in_ch * out_ch + 50_000
        state, cycles = app.run(max_cycles=max_cyc)

        actual = read_output(state, out_ch)
        expected = reference_conv(weights, input_chw).reshape(out_ch, SPATIAL)

    mismatches = compare(actual, expected)
    return cycles, mismatches


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        inst_file = Path(tmp) / "conv_8x8.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(inst_file))

        header = (
            f"{'config':>16} {'cycles':>10} {'MAC/cyc':>8} "
            f"{'theory':>10} {'eff%':>6} {'corr':>5} {'time(s)':>8}"
        )
        print(header)
        print("-" * len(header))

        total_cycles = 0
        total_theory = 0
        all_correct = True

        for in_ch, out_ch in CONFIGS:
            t0 = time.time()
            cycles, mismatches = run_config(inst_file, in_ch, out_ch)
            elapsed = time.time() - t0

            # Theoretical MULT-issue floor: 9 taps * 64 spatial * in_ch * out_ch / 128
            theory = 9 * SPATIAL * in_ch * out_ch // 128
            mac_per_cyc = (9 * SPATIAL * in_ch * out_ch) / cycles
            efficiency = theory / cycles * 100

            ok = mismatches == 0
            all_correct = all_correct and ok

            label = f"ic={in_ch} oc={out_ch}"
            print(
                f"{label:>16} {cycles:>10} {mac_per_cyc:>8.2f} "
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
