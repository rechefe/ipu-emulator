"""Benchmark: conv_first_layer (256x256x3 -> 128x128x16, stride-2).

Single fixed MobileViT-S stem configuration.

Run from repo root:
    PYTHONPATH=src/tools/ipu-emu-py/src:src/tools/ipu-common/src:src/tools/ipu-apps/src:src/tools/ipu-as-py/src \
        python -m ipu_apps.convolutions_universal.benchmarks.bench_conv_first_layer
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from ipu_apps.convolutions_universal.conv_first_layer import (
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
from ipu_apps.convolutions_universal.profiling._utils import assemble_if_needed
from ipu_emu.ipu_math import DType, ipu_mult, ipu_add
from ipu_apps.convolutions_universal.benchmarks._utils import (
    gen_int8,
    check_correct,
    print_table_header,
    print_table_row,
    print_table_footer,
)

ASM = (
    Path(__file__).resolve().parents[1]
    / "conv_first_layer"
    / "conv_first_layer.asm"
)


def _reference(input_bytes: bytes, kernel_bytes: bytes) -> np.ndarray:
    """Exact reference matching the IPU's half-plane boundary handling.

    Input:  CHW layout, 256x256x3.
    Kernel: f*27 + c*9 + tap.
    Output: (OUT_ROWS, OUT_CHANNELS, OUT_COLS) int8 array.

    The assembly processes left half (cols 0..127) and right half (cols 128..255)
    independently — kc=-1 at the start of each half is zeroed (left-border mask).
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
    bin_path = assemble_if_needed(ASM)
    print_table_header()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        rng = np.random.RandomState(0)
        input_arr = rng.randint(-3, 4, size=IN_CHANNELS * IN_ROWS * IN_COLS, dtype=np.int8)
        kernel_arr = rng.randint(-3, 4, size=OUT_CHANNELS * IN_CHANNELS * 9, dtype=np.int8)

        input_bytes = input_arr.view(np.uint8).tobytes()
        kernel_bytes = kernel_arr.view(np.uint8).tobytes()

        inp_f = tmp / "input.bin"
        krn_f = tmp / "kernel.bin"
        inp_f.write_bytes(input_bytes)
        krn_f.write_bytes(kernel_bytes)

        app = ConvFirstLayerApp(
            inst_path=bin_path,
            input_path=inp_f,
            kernel_path=krn_f,
            output_path=None,
        )
        state, cycles = app.run(max_cycles=200_000_000)

        # Read output: (OUT_ROWS * OUT_CHANNELS) x 128 bytes
        total_bytes = OUT_ROWS * OUT_CHANNELS * OUT_COLS
        raw = state.xmem.read_address(OUTPUT_BASE_ADDR, total_bytes)
        # Layout: (row * OUT_CHANNELS + filter) * OUT_COLS
        actual = np.frombuffer(raw, dtype=np.uint8).view(np.int8)
        actual = actual.reshape(OUT_ROWS, OUT_CHANNELS, OUT_COLS)

        print("  Computing reference (slow) ...", flush=True)
        expected = _reference(input_bytes, kernel_bytes)

        name = "256x256x3->128x128x16 stride-2"
        correct = check_correct(actual, expected, name)

        macs = OUT_ROWS * OUT_COLS * IN_CHANNELS * OUT_CHANNELS * 9
        print_table_row(name, cycles, macs, correct)

    print_table_footer("conv_first_layer")


if __name__ == "__main__":
    main()
