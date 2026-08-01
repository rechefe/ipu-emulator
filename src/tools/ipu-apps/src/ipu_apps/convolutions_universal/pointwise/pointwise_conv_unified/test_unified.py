"""Standalone correctness test for pointwise_conv_unified.

Assembles the asm fresh each run (no persisted binary) and compares output
against a numpy reference. Designed to be runnable directly:

    PYTHONPATH=... python -m ipu_apps.convolutions_universal.pointwise.pointwise_conv_unified.test_unified
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.convolutions_universal.pointwise.pointwise_conv_unified import (
    PointwiseConvUnifiedApp,
    OUTPUT_BASE_ADDR,
)


ASM_PATH = Path(__file__).resolve().parent / "pointwise_conv_unified.asm"

TEST_CONFIGS = [
    # Single-pass cases (in_ch <= 128)
    (16, 16,   8,  8),
    (16, 16,  16,  8),
    (16, 16,  24,  8),
    (16, 16,  32,  8),
    (16, 16,  40,  8),
    (16, 16,  56,  8),
    (16, 16,  64,  8),
    (16, 16,  80,  8),
    (16, 16,  96,  8),
    (16, 16, 112,  8),
    (16, 16, 128,  8),
    # Multi-pass cases (in_ch > 128)
    (16, 16, 144,  8),  # 1 full + tail 16
    (16, 16, 160,  8),  # 1 full + tail 32
    (16, 16, 192,  8),  # 1 full + tail 64
    (16, 16, 224,  8),  # 1 full + tail 96
    (16, 16, 240,  8),  # 1 full + tail 112
    (16, 16, 256,  8),  # 2 full passes
    (16, 16, 384,  8),  # 3 full passes
    (16, 16, 400,  8),  # 3 full + tail 16
    # Larger spatial / out_ch variety
    (32, 32, 128, 16),
    (32, 32, 144, 16),
    (32, 32,  96, 32),
    (32, 32, 192, 32),
    (64, 64, 128,  4),
    (64, 64, 128, 64),
]


def pack_input(input_chw: np.ndarray, rows: int, cols: int) -> bytes:
    """Pack CHW int8 input as row_groups of 128B per channel."""
    channels, _, _ = input_chw.shape
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


def reference_pointwise(
    weights: np.ndarray, input_chw: np.ndarray
) -> np.ndarray:
    """Reference pointwise conv. Returns (out_ch, rows, cols) int8."""
    # weights: (out_ch, in_ch), input_chw: (in_ch, rows, cols)
    inp32 = input_chw.astype(np.int32)
    w32 = weights.astype(np.int32)
    result = np.einsum("oi,ihw->ohw", w32, inp32)
    return result.clip(-128, 127).astype(np.int8)


def read_output(state, rows, cols, out_ch) -> np.ndarray:
    """Read OC×row_group blocks of 128B from output region."""
    rows_per_chunk = 128 // cols
    row_groups = (rows * cols) // 128
    raw = state.xmem.read_address(OUTPUT_BASE_ADDR, row_groups * out_ch * 128)
    vals = np.frombuffer(raw, dtype=np.uint8).reshape(row_groups, out_ch, 128)
    result = np.empty((out_ch, rows, cols), dtype=np.int8)
    for rg in range(row_groups):
        for oc in range(out_ch):
            block = vals[rg, oc, :rows_per_chunk * cols].view(np.int8)
            result[oc, rg * rows_per_chunk:(rg + 1) * rows_per_chunk, :] = (
                block.reshape(rows_per_chunk, cols)
            )
    return result


def run_one(inst_file: Path, rows: int, cols: int, in_ch: int, out_ch: int):
    rng = np.random.RandomState(42 + in_ch * 7 + out_ch + rows + cols)
    weights = rng.randint(-3, 4, size=(out_ch, in_ch), dtype=np.int8)
    input_chw = rng.randint(-3, 4, size=(in_ch, rows, cols), dtype=np.int8)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        input_file = tmp / "input.bin"
        kernel_file = tmp / "kernel.bin"
        input_file.write_bytes(pack_input(input_chw, rows, cols))
        kernel_file.write_bytes(
            weights.reshape(out_ch * in_ch).view(np.uint8).tobytes()
        )

        app = PointwiseConvUnifiedApp(
            inst_path=inst_file,
            input_path=input_file,
            kernel_path=kernel_file,
            output_path=None,
            dtype="INT8",
            rows=rows, cols=cols, in_channels=in_ch, out_channels=out_ch,
        )
        max_cyc = 50 * in_ch * out_ch * (rows * cols // 128) + 100_000
        state, cycles = app.run(max_cycles=max_cyc)

        actual = read_output(state, rows, cols, out_ch)
        expected = reference_pointwise(weights, input_chw)

    mismatches = int(np.sum(actual != expected))
    return cycles, mismatches, actual, expected


def main() -> None:
    template_src = ASM_PATH.read_text()

    with tempfile.TemporaryDirectory() as tmp:
        inst_file = Path(tmp) / "unified.bin"
        print("Assembling pointwise_conv_unified.asm ...", flush=True)
        assemble_to_bin_file(template_src, str(inst_file))

        print(f"\n{'config':>26} {'cycles':>10} {'mismatch':>10} {'status':>8}")
        print("-" * 60)

        all_ok = True
        for rows, cols, in_ch, out_ch in TEST_CONFIGS:
            label = f"{rows}x{cols} ic={in_ch} oc={out_ch}"
            try:
                cycles, mm, actual, expected = run_one(
                    inst_file, rows, cols, in_ch, out_ch
                )
                ok = mm == 0
                status = "PASS" if ok else "FAIL"
                print(f"{label:>26} {cycles:>10} {mm:>10} {status:>8}")
                if not ok:
                    all_ok = False
                    print(f"  first OC actual: {actual[0, 0, :8]}")
                    print(f"  first OC expect: {expected[0, 0, :8]}")
            except Exception as e:
                all_ok = False
                print(f"{label:>26} {'ERROR':>10} {'-':>10} {'FAIL':>8}")
                print(f"  {type(e).__name__}: {e}")

        print("-" * 60)
        print(f"Overall: {'PASS' if all_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
