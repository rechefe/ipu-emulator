"""Benchmark conv_universal across 10 configs: cycles + real MULT utilization.

Runs each config through the emulator, compares output to a real PyTorch
conv2d reference for correctness (tolerance-based -- IPU FP32 accumulation
order differs from PyTorch's), and reports cycles and MULT-slot utilization
read directly from ``state.stats`` (the emulator's live per-cycle occupancy
counter -- not an analytical estimate).

Usage::

    PYTHONPATH=... python -m ipu_apps.convolutions_universal.conv.conv_universal.benchmark.benchmark
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.convolutions_universal.conv.conv_universal import ConvUniversalApp
from ipu_apps.convolutions_universal import unpack_output_chunked
from ipu_apps.convolutions_universal.benchmarking import BenchRow, print_and_write_table


ASM_PATH = Path(__file__).resolve().parents[1] / "conv_universal.asm"

_TOL = 1e-2

# (height, width, in_ch, out_ch) — 10 configs covering shapes, channel
# counts, partial vs full last blocks, and small/large spatial sizes.
CONFIGS = [
    (16, 16, 28, 4),     # exactly 1 full block, smallest spatial
    (16, 16, 16, 4),     # partial last block (16 < 28)
    (16, 16, 56, 4),     # exactly 2 full blocks
    (32, 32, 28, 8),     # 1 full block, multi-chunk
    (32, 32, 32, 16),    # partial last block, multi-filter
    (32, 32, 16, 32),    # partial last block, more filters than channels
    (64, 64, 32, 32),    # primary benchmark — large
    (64, 64, 28, 28),    # 1 full block, larger spatial
    (32, 32, 64, 8),     # large in_ch, multiple full+partial blocks
    (16, 16, 84, 4),     # exactly 3 full blocks
]


def reference_conv(weights: np.ndarray, input_chw: np.ndarray) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(input_chw).unsqueeze(0)
    w = torch.from_numpy(weights)
    return F.conv2d(x, w, padding=1).squeeze(0).numpy()


def run_config(inst_file: Path, height: int, width: int, in_ch: int, out_ch: int):
    rng = np.random.RandomState(42 + in_ch * 7 + out_ch * 13 + height + width)
    weights = (rng.randn(out_ch, in_ch, 3, 3) * 0.2).astype(np.float32)
    input_chw = (rng.randn(in_ch, height, width) * 0.5).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        input_file = tmp / "input.bin"
        input_file.write_bytes(input_chw.tobytes())

        app = ConvUniversalApp(
            inst_path=inst_file,
            input_path=input_file,
            kernel=weights,
            output_path=None,
            height=height, width=width,
            in_channels=in_ch, out_channels=out_ch,
        )

        max_cyc = 2000 * app.num_chunks * out_ch * app.blocks_per_filter + 50_000
        state, cycles = app.run(max_cycles=max_cyc)

        total_elements = app.num_chunks * out_ch * 128
        raw = state.xmem.read_address(app.output_base_addr, total_elements * 4)
        padded_out = unpack_output_chunked(raw, out_ch, app.rows, app.cols)
        actual = padded_out[:, :height, :width]
        expected = reference_conv(weights, input_chw)

    max_diff = float(np.abs(actual - expected).max())
    return cycles, max_diff, state.stats.mult_utilization


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        inst_file = Path(tmp) / "conv_universal.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(inst_file))

        rows_out = []
        for height, width, in_ch, out_ch in CONFIGS:
            t0 = time.time()
            cycles, max_diff, mult_util = run_config(
                inst_file, height, width, in_ch, out_ch
            )
            elapsed = time.time() - t0
            rows_out.append(BenchRow(
                label=f"{height}x{width}x{in_ch}x{out_ch}",
                cycles=cycles,
                mult_utilization=mult_util,
                correct=(max_diff < _TOL),
                elapsed_s=elapsed,
            ))

        out_path = Path(__file__).resolve().parent / "results.md"
        print_and_write_table("conv_universal", rows_out, out_path)


if __name__ == "__main__":
    main()
