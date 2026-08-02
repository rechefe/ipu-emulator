"""Benchmark residual_add: cycles + real MULT util.

Element-wise FP32 add of two tensors (one channel per 512-byte wide chunk).
Reports cycles and MULT-slot utilization read directly from ``state.stats``.

Usage::

    PYTHONPATH=... python -m ipu_apps.convolutions_universal.residual_add.benchmark.benchmark
"""

from __future__ import annotations

import struct
import tempfile
import time
from pathlib import Path

import numpy as np

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.convolutions_universal.residual_add import (
    ResidualAddApp,
    OUTPUT_BASE,
    LANES_PER_CHUNK,
    WIDE_CHUNK_BYTES,
)
from ipu_apps.convolutions_universal.benchmarking import BenchRow, print_and_write_table


ASM_PATH = Path(__file__).resolve().parents[1] / "residual_add.asm"

# MobileViT S residual sizes.
CONFIGS = [64, 96, 128, 160]


def pack_fp32(values: np.ndarray, num_channels: int) -> bytes:
    packed = bytearray(num_channels * WIDE_CHUNK_BYTES)
    flat = values.astype("<f4").reshape(num_channels, LANES_PER_CHUNK)
    for ch in range(num_channels):
        struct.pack_into(f"<{LANES_PER_CHUNK}f", packed, ch * WIDE_CHUNK_BYTES, *flat[ch])
    return bytes(packed)


def run_config(inst_file: Path, num_ch: int):
    rng = np.random.RandomState(42 + num_ch)
    shape = (num_ch, LANES_PER_CHUNK)
    a = rng.uniform(-4.0, 4.0, size=shape).astype(np.float32)
    b = rng.uniform(-6.0, 6.0, size=shape).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        a_file = tmp / "input_a.bin"
        b_file = tmp / "input_b.bin"
        a_file.write_bytes(pack_fp32(a, num_ch))
        b_file.write_bytes(pack_fp32(b, num_ch))

        app = ResidualAddApp(
            inst_path=inst_file,
            input_a_path=a_file,
            input_b_path=b_file,
            output_path=None,
            num_channels=num_ch,
        )
        max_cyc = num_ch * 10 + 1000
        state, cycles = app.run(max_cycles=max_cyc)

        total_output = num_ch * WIDE_CHUNK_BYTES
        actual = state.xmem.read_address(OUTPUT_BASE, total_output)

    expected = (a + b).astype(np.float32)
    mismatches = 0
    for ch in range(num_ch):
        for lane in range(LANES_PER_CHUNK):
            idx = ch * LANES_PER_CHUNK + lane
            act_val = struct.unpack_from("<f", actual, idx * 4)[0]
            if act_val != float(expected[ch, lane]):
                mismatches += 1

    return cycles, mismatches, state.stats.mult_utilization


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        inst_file = Path(tmp) / "residual_add.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(inst_file))

        rows_out = []
        for num_ch in CONFIGS:
            t0 = time.time()
            cycles, mismatches, mult_util = run_config(inst_file, num_ch)
            elapsed = time.time() - t0
            rows_out.append(BenchRow(
                label=f"channels={num_ch}",
                cycles=cycles,
                mult_utilization=mult_util,
                correct=(mismatches == 0),
                elapsed_s=elapsed,
            ))

        out_path = Path(__file__).resolve().parent / "results.md"
        print_and_write_table("residual_add", rows_out, out_path)


if __name__ == "__main__":
    main()
