"""Benchmark conv_universal_wide384: cycles + real MULT util.

Single-stage app (unlike the stride2 apps, no two-stage stats-composition
workaround needed) -- ``state.stats.mult_utilization`` is read directly.

Usage::

    PYTHONPATH=... python -m ipu_apps.convolutions_universal.conv.conv_universal_wide384.benchmark.benchmark
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ipu_emu.ipu_math import DType, ipu_mult, ipu_add

from ipu_apps.convolutions_universal.conv.conv_universal_wide384 import (
    ConvUniversalWide384App,
)
from ipu_apps.convolutions_universal.benchmarking import BenchRow, print_and_write_table


# (width, rows, in_channels, out_channels) -- width multiple of 128, >=384;
# out_channels even (app requirement).
CONFIGS = [
    (384, 8, 1, 2),     # minimal cpr=3 case
    (384, 16, 3, 4),    # moderate spatial + channels
    (512, 8, 2, 4),     # cpr=4
    (384, 8, 16, 4),    # in_channels > FPB=14: exercises kernel-reload path
    (640, 4, 1, 2),     # cpr=5, wider than the primary target
]


def _pack_input_wide384(input_chw: np.ndarray, width: int) -> bytes:
    in_ch, rows, w = input_chw.shape
    assert w == width
    cpr = width // 128
    packed = bytearray(rows * in_ch * cpr * 128)
    for r in range(rows):
        for ic in range(in_ch):
            row_base = (r * in_ch + ic) * cpr * 128
            for cc in range(cpr):
                for lane in range(128):
                    c = cc * 128 + lane
                    packed[row_base + cc * 128 + lane] = np.uint8(
                        input_chw[ic, r, c]
                    ).item()
    return bytes(packed)


def reference_conv_wide384(
    weights: np.ndarray, input_chw: np.ndarray, rows: int, width: int,
) -> bytes:
    out_ch, in_ch, _, _ = weights.shape
    cpr = width // 128
    output = bytearray(rows * out_ch * cpr * 128)
    for r in range(rows):
        for f in range(out_ch):
            row_base = (r * out_ch + f) * cpr * 128
            for c in range(width):
                acc = 0
                for ic in range(in_ch):
                    for dr in range(3):
                        for dc in range(3):
                            nr, nc = r + dr - 1, c + dc - 1
                            if 0 <= nr < rows and 0 <= nc < width:
                                a = int(weights[f, ic, dr, dc])
                                b = int(input_chw[ic, nr, nc])
                                prod = ipu_mult(a, b, DType.INT8)
                                acc = ipu_add(acc, prod, DType.INT8)
                clamped = max(-128, min(127, acc))
                cc = c // 128
                lane = c % 128
                output[row_base + cc * 128 + lane] = clamped & 0xFF
    return bytes(output)


def compare(actual: bytes, expected: bytes) -> int:
    return sum(1 for a, e in zip(actual, expected) if a != e)


def run_config(width: int, rows: int, in_ch: int, out_ch: int, seed: int):
    rng = np.random.RandomState(seed)
    weights = rng.randint(-32, 33, size=(out_ch, in_ch, 3, 3), dtype=np.int8)
    input_chw = rng.randint(-32, 33, size=(in_ch, rows, width), dtype=np.int8)

    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="conv_wide384_bench_"))
    input_file = tmp / "input.bin"
    input_file.write_bytes(_pack_input_wide384(input_chw, width))

    app = ConvUniversalWide384App(
        input_path=input_file,
        kernel=weights,
        output_path=None,
        width=width,
        rows=rows,
        in_channels=in_ch,
        out_channels=out_ch,
    )
    cpr = width // 128
    max_cyc = 200 * rows * out_ch * cpr * in_ch * 9 + 50_000
    state, cycles = app.run(max_cycles=max_cyc)

    total_bytes = rows * out_ch * cpr * 128
    actual = state.xmem.read_address(0x100000, total_bytes)
    expected = reference_conv_wide384(weights, input_chw, rows, width)

    return cycles, compare(actual, expected), state.stats.mult_utilization


def main() -> None:
    rows_out = []
    for width, rows, in_ch, out_ch in CONFIGS:
        seed = 100 + width + rows * 7 + in_ch * 13 + out_ch
        t0 = time.time()
        cycles, mismatches, mult_util = run_config(width, rows, in_ch, out_ch, seed)
        elapsed = time.time() - t0
        rows_out.append(BenchRow(
            label=f"{width}x{rows}x{in_ch}x{out_ch}",
            cycles=cycles,
            mult_utilization=mult_util,
            correct=(mismatches == 0),
            elapsed_s=elapsed,
        ))

    out_path = Path(__file__).resolve().parent / "results.md"
    print_and_write_table("conv_universal_wide384", rows_out, out_path)


if __name__ == "__main__":
    main()
