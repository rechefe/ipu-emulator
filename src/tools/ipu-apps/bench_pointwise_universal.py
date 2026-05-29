"""Benchmark: universal pointwise conv on 4 MobileViT-S configurations.

Runs PointwiseConvUniversalApp on four representative pointwise layers from
MobileViT-S and prints cycle counts.

Usage (from repo root):
    PYTHONPATH=src/tools/ipu-emu-py/src:src/tools/ipu-common/src:src/tools/ipu-apps/src:src/tools/ipu-as-py/src \
        python src/tools/ipu-apps/bench_pointwise_universal.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from ipu_as.lark_tree import assemble_to_bin_file
from ipu_apps.convolutions_universal.pointwise.pointwise_conv_universal import (
    PointwiseConvUniversalApp,
)

ASM_PATH = (
    Path(__file__).resolve().parent
    / "src"
    / "ipu_apps"
    / "convolutions_universal"
    / "pointwise_conv_universal"
    / "pointwise_conv_universal.asm"
)

BIN_PATH = ASM_PATH.with_suffix(".bin")

# Four MobileViT-S pointwise configurations (from REQUIRED_CONVOLUTIONS.md)
CONFIGS = [
    # name,                        rows, cols, in_ch, out_ch
    ("Expand-1  128x128 16->64",   128,  128,   16,    64),
    ("Expand-3   64x64 64->256",    64,   64,   64,   256),
    ("Expand-4   32x32 96->384",    32,   32,   96,   384),
    ("Expand-5   16x16 128->512",   16,   16,  128,   512),
]


def _pack_input(input_raw: bytes, rows: int, cols: int, in_channels: int) -> bytes:
    """Pack raw input into the multi-channel layout expected by the assembly.

    Raw layout: input_raw[ch * rows * cols + spatial_pos]  (channel-first, row-major)

    Packed layout (multi-channel memory layout):
      row_group × in_channels × 128 bytes
      Each 128-byte block holds one channel's data for that row-group.
      rows_per_chunk = 128 // cols  (spatial rows packed per 128-byte chunk)
    """
    spatial = rows * cols
    rows_per_chunk = 128 // cols
    row_groups = (rows * cols) // 128
    packed = bytearray(row_groups * in_channels * 128)

    for rg in range(row_groups):
        for ch in range(in_channels):
            dst = (rg * in_channels + ch) * 128
            # Copy rows_per_chunk rows of this channel into one 128-byte block
            for r in range(rows_per_chunk):
                spatial_row = rg * rows_per_chunk + r
                src = ch * spatial + spatial_row * cols
                packed[dst + r * cols : dst + r * cols + cols] = input_raw[src : src + cols]

    return bytes(packed)


def _gen_data(rows: int, cols: int, in_ch: int, out_ch: int, seed: int = 0):
    """Generate random INT8 input (packed) + kernel bytes (raw layout)."""
    rng = np.random.RandomState(seed)
    spatial = rows * cols
    input_raw = rng.randint(-3, 4, size=in_ch * spatial, dtype=np.int8).view(np.uint8).tobytes()
    kernel_raw = rng.randint(-3, 4, size=out_ch * in_ch, dtype=np.int8).view(np.uint8).tobytes()
    input_packed = _pack_input(input_raw, rows, cols, in_ch)
    return input_packed, kernel_raw


def main() -> None:
    # Assemble once (or reuse pre-built binary)
    if not BIN_PATH.exists():
        print(f"Assembling {ASM_PATH.name} ...", flush=True)
        assemble_to_bin_file(ASM_PATH.read_text(), str(BIN_PATH))
        print(f"  -> {BIN_PATH}")

    print()
    print(f"{'Configuration':<35} {'rows':>4} {'cols':>4} {'in_ch':>6} {'out_ch':>7}  {'cycles':>10}   ops/cycle")
    print("-" * 82)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name, rows, cols, in_ch, out_ch in CONFIGS:
            input_packed, kernel_raw = _gen_data(rows, cols, in_ch, out_ch)

            input_file  = tmp / f"input_{name.split()[0]}.bin"
            kernel_file = tmp / f"kernel_{name.split()[0]}.bin"
            input_file.write_bytes(input_packed)
            kernel_file.write_bytes(kernel_raw)

            app = PointwiseConvUniversalApp(
                inst_path=BIN_PATH,
                input_path=input_file,
                kernel_path=kernel_file,
                output_path=None,
                dtype="INT8",
                rows=rows,
                cols=cols,
                in_channels=in_ch,
                out_channels=out_ch,
            )
            _state, cycles = app.run(max_cycles=50_000_000)

            ops = 2 * rows * cols * in_ch * out_ch  # multiply-adds × 2
            ops_per_cycle = ops / cycles if cycles else 0.0

            print(
                f"{name:<35} {rows:>4} {cols:>4} {in_ch:>6} {out_ch:>7}  "
                f"{cycles:>10,}   {ops_per_cycle:.2f}"
            )


if __name__ == "__main__":
    main()
