"""Benchmark pointwise_conv_universal across configs: cycles + correctness.

Runs each config through the emulator, compares output to a numpy reference
for correctness, and reports cycles, theoretical minimum cycles
(num_row_groups * out_channels * in_channels cyc/OC — absolute MULT floor),
and end-to-end cyc/OC.

For pointwise conv, the MULT floor is 1 cyc/ic/OC (one multiply per
input-channel per output-channel per row-group). The pipeline processes
4 ICs per 4 cycles (body), plus a fixed 4-cycle epilogue per OC, plus
per-OC overhead (guard, store). Achievable efficiency depends on G and in_ch.

Usage::

    PYTHONPATH=... python -m ipu_apps.convolutions_universal.pointwise_conv_universal.benchmark.benchmark
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.convolutions_universal.pointwise_conv_universal import (
    PointwiseConvUniversalApp,
    OUTPUT_BASE_ADDR,
)
from ipu_apps.convolutions_universal.pointwise_conv_universal.gen_test_data import (
    reference_pointwise_conv,
)
from ipu_emu.ipu_math import DType


ASM_PATH = Path(__file__).resolve().parents[1] / "pointwise_conv_universal.asm"


# (rows, cols, in_channels, out_channels) — 2 configs per spatial size
CONFIGS = [
    (128, 128,  16,  64),  # 128x128
    (128, 128,  64,  32),  # 128x128
    ( 64,  64, 128,  64),  # 64x64
    ( 64,  64,  64, 256),  # 64x64
    ( 32,  32, 256,  96),  # 32x32
    ( 32,  32,  96, 144),  # 32x32
    ( 16,  16, 384, 128),  # 16x16
    ( 16,  16, 128, 192),  # 16x16
]


def compare(actual: bytes, expected: bytes) -> int:
    mismatches = 0
    for i in range(len(expected)):
        a = actual[i]
        e = expected[i]
        if a != e:
            mismatches += 1
    return mismatches


def run_config(inst_file: Path, rows: int, cols: int, in_ch: int, out_ch: int):
    rng = np.random.RandomState(42 + in_ch * 7 + out_ch + rows + cols)
    row_groups = (rows * cols) // 128
    input_raw = rng.randint(-8, 9, size=row_groups * in_ch * 128, dtype=np.int8)
    input_bytes = input_raw.view(np.uint8).tobytes()
    kernel_raw = rng.randint(-4, 5, size=in_ch * out_ch, dtype=np.int8)
    kernel_bytes = kernel_raw.view(np.uint8).tobytes()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        inp = tmp / "input.bin"
        inp.write_bytes(input_bytes)
        ker = tmp / "kernel.bin"
        ker.write_bytes(kernel_bytes)

        app = PointwiseConvUniversalApp(
            inst_path=inst_file,
            input_path=inp,
            kernel_path=ker,
            output_path=None,
            dtype="INT8",
            rows=rows,
            cols=cols,
            in_channels=in_ch,
            out_channels=out_ch,
        )

        # Budget: generous ceiling based on config size
        max_cyc = row_groups * out_ch * in_ch * 4 + 200_000
        state, cycles = app.run(max_cycles=max_cyc)

        # Output is now 128 bytes (int8) per output-channel per row-group
        total_bytes = row_groups * out_ch * 128
        actual = state.xmem.read_address(OUTPUT_BASE_ADDR, total_bytes)
        expected = reference_pointwise_conv(
            input_bytes, kernel_bytes, DType.INT8, rows, cols, in_ch, out_ch
        )

    mismatches = compare(actual, expected)
    return cycles, mismatches, row_groups


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        inst_file = Path(tmp) / "pointwise_conv_universal.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(inst_file))

        header = (
            f"{'config':>24} {'cycles':>10} {'cyc/OC':>8} "
            f"{'theory':>10} {'eff%':>6} {'corr':>5} {'time(s)':>8}"
        )
        print(header)
        print("-" * len(header))

        total_cycles = 0
        total_theory = 0
        all_correct = True

        for rows, cols, in_ch, out_ch in CONFIGS:
            t0 = time.time()
            cycles, mismatches, row_groups = run_config(
                inst_file, rows, cols, in_ch, out_ch
            )
            elapsed = time.time() - t0

            # Theoretical min: 1 cyc per multiply tap = in_ch cyc per OC per row-group
            total_oc = row_groups * out_ch
            theory = in_ch * total_oc
            cyc_per_oc = cycles / total_oc
            efficiency = theory / cycles * 100

            ok = mismatches == 0
            all_correct = all_correct and ok

            label = f"{rows}x{cols} in={in_ch} out={out_ch}"
            print(
                f"{label:>24} {cycles:>10} {cyc_per_oc:>8.2f} "
                f"{theory:>10} {efficiency:>5.1f}% "
                f"{'PASS' if ok else 'FAIL':>5} {elapsed:>8.2f}"
            )

            total_cycles += cycles
            total_theory += theory

        print("-" * len(header))
        print(
            f"{'TOTAL':>24} {total_cycles:>10} {'':>8} "
            f"{total_theory:>10} {total_theory / total_cycles * 100:>5.1f}% "
            f"{'PASS' if all_correct else 'FAIL':>5}"
        )


if __name__ == "__main__":
    main()
