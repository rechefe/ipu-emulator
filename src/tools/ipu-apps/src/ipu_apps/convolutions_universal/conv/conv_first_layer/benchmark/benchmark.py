"""Benchmark conv_first_layer: 256x256x3 -> 128x128x16, stride-2.

Single fixed configuration. Output is INT8 (quantized/clamped to [-128, 127]).
The reference matches the IPU's half-plane boundary handling: left border mask
zeroes kc=-1 at col 0 of each half (cols 0-127 and cols 128-255 processed
independently, so the half-boundary is not zero-padded).

Reports cycles, theoretical minimum (9 * OUT_ROWS * OUT_COLS * IN_CH * OUT_CH / 128
— MULT-issue floor at 128 mults/cycle), and efficiency.

Usage::

    PYTHONPATH=... python -m ipu_apps.convolutions_universal.conv.conv_first_layer.benchmark.benchmark
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.convolutions_universal.conv.conv_first_layer import (
    ConvFirstLayerApp,
    OUTPUT_BASE_ADDR,
    IN_ROWS,
    IN_COLS,
    IN_CHANNELS,
    OUT_ROWS,
    OUT_COLS,
    OUT_CHANNELS,
    CHANNEL_STRIDE,
)
from ipu_emu.ipu_math import DType, ipu_mult, ipu_add


ASM_PATH = Path(__file__).resolve().parents[1] / "conv_first_layer.asm"


def reference(input_bytes: bytes, kernel_bytes: bytes) -> np.ndarray:
    """Exact reference matching the IPU's half-plane boundary handling.

    The assembly processes left half (cols 0-127) and right half (cols 128-255)
    independently — kc=-1 at the start of each half is zeroed (left-border mask).
    Returns (OUT_ROWS, OUT_CHANNELS, OUT_COLS) int8 array.
    """
    dtype = DType.INT8
    output = np.zeros((OUT_ROWS, OUT_CHANNELS, OUT_COLS), dtype=np.int8)

    for f in range(OUT_CHANNELS):
        for out_r in range(OUT_ROWS):
            center_r = out_r * 2
            for out_c in range(OUT_COLS):
                center_c = out_c * 2
                half_start = 0 if out_c < 64 else 128

                acc: int = 0
                for c in range(IN_CHANNELS):
                    for kr in range(-1, 2):
                        for kc in range(-1, 2):
                            ir = center_r + kr
                            ic = center_c + kc
                            if ir < 0 or ir >= IN_ROWS:
                                continue
                            if ic < half_start or ic >= half_start + 128:
                                continue
                            in_idx = c * CHANNEL_STRIDE + ir * IN_COLS + ic
                            b = input_bytes[in_idx]
                            tap = (kr + 1) * 3 + (kc + 1)
                            a = kernel_bytes[f * 27 + c * 9 + tap]
                            prod = ipu_mult(a, b, dtype)
                            acc = ipu_add(acc, prod, dtype)

                output[out_r, f, out_c] = max(-128, min(127, acc))

    return output


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        inst_file = Path(tmp) / "conv_first_layer.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(inst_file))

        header = (
            f"{'config':>36} {'cycles':>10} {'MAC/cyc':>8} "
            f"{'theory':>10} {'eff%':>6} {'corr':>5} {'time(s)':>8}"
        )
        print(header)
        print("-" * len(header))

        rng = np.random.RandomState(42)
        input_arr = rng.randint(-32, 33, size=IN_CHANNELS * IN_ROWS * IN_COLS, dtype=np.int8)
        kernel_arr = rng.randint(-32, 33, size=OUT_CHANNELS * IN_CHANNELS * 9, dtype=np.int8)
        input_bytes = input_arr.view(np.uint8).tobytes()
        kernel_bytes = kernel_arr.view(np.uint8).tobytes()

        with tempfile.TemporaryDirectory() as dtmp:
            dtmp = Path(dtmp)
            inp_f = dtmp / "input.bin"
            krn_f = dtmp / "kernel.bin"
            inp_f.write_bytes(input_bytes)
            krn_f.write_bytes(kernel_bytes)

            app = ConvFirstLayerApp(
                inst_path=inst_file,
                input_path=inp_f,
                kernel_path=krn_f,
                output_path=None,
            )

            t0 = time.time()
            state, cycles = app.run(max_cycles=500_000_000)
            elapsed = time.time() - t0

        # Read output: (OUT_ROWS * OUT_CHANNELS) x OUT_COLS bytes
        total_bytes = OUT_ROWS * OUT_CHANNELS * OUT_COLS
        raw = state.xmem.read_address(OUTPUT_BASE_ADDR, total_bytes)
        actual = np.frombuffer(raw, dtype=np.uint8).view(np.int8).reshape(
            OUT_ROWS, OUT_CHANNELS, OUT_COLS
        )

        print("  Computing reference (slow) ...", flush=True)
        t_ref = time.time()
        expected = reference(input_bytes, kernel_bytes)
        print(f"  Reference done in {time.time() - t_ref:.1f}s", flush=True)

        mismatches = int(np.sum(actual != expected))
        ok = mismatches == 0

        theory = 9 * OUT_ROWS * OUT_COLS * IN_CHANNELS * OUT_CHANNELS // 128
        mac_per_cyc = (9 * OUT_ROWS * OUT_COLS * IN_CHANNELS * OUT_CHANNELS) / cycles
        efficiency = theory / cycles * 100

        name = f"256x256x{IN_CHANNELS}->{OUT_ROWS}x{OUT_COLS}x{OUT_CHANNELS} stride-2"
        print(
            f"{name:>36} {cycles:>10} {mac_per_cyc:>8.2f} "
            f"{theory:>10} {efficiency:>5.1f}% "
            f"{'PASS' if ok else 'FAIL':>5} {elapsed:>8.2f}"
        )
        print("-" * len(header))
        print(f"\n  Speed-of-light: 128 MAC/cycle  |  conv_first_layer")


if __name__ == "__main__":
    main()
