"""Debug runner for the first-layer conv app (256x256x3 -> 128x128x16, stride 2, FP32).

Assembles the .asm fresh, runs a random FP32 batch, and prints the cycle
count and max error against a real PyTorch pad=1 stride-2 conv + bias + ReLU
reference.

Usage::

    PYTHONPATH=... python -m ipu_apps.convolutions_universal.conv.conv_first_layer
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.convolutions_universal.conv.conv_first_layer import (
    ConvFirstLayerApp,
    OUTPUT_BASE_ADDR,
    OUT_ROWS,
    OUT_COLS,
    OUT_CHANNELS,
    IN_ROWS,
    IN_CHANNELS,
    IN_COLS,
)
from ipu_apps.convolutions_universal.conv.conv_first_layer.test_conv_first_layer import (
    reference,
)

ASM_PATH = ConvFirstLayerApp.ASM_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Run first-layer conv with a PyTorch cross-check")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-cycles", type=int, default=500_000_000)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    x = (rng.randn(IN_CHANNELS, IN_ROWS, IN_COLS) * 0.5).astype(np.float32)
    k = (rng.randn(OUT_CHANNELS, IN_CHANNELS, 3, 3) * 0.2).astype(np.float32)
    b = (rng.randn(OUT_CHANNELS) * 0.3).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        inst = tmp / "conv_first_layer.bin"
        infile = tmp / "input.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(inst))
        infile.write_bytes(x.tobytes())

        app = ConvFirstLayerApp(
            inst_path=inst, input_path=infile, kernel=k, bias=b, output_path=None
        )
        state, cycles = app.run(max_cycles=args.max_cycles)
        total_elements = OUT_ROWS * OUT_CHANNELS * OUT_COLS
        raw = state.xmem.read_address(OUTPUT_BASE_ADDR, total_elements * 4)
        out = np.frombuffer(raw, dtype=np.float32).reshape(OUT_ROWS, OUT_CHANNELS, OUT_COLS)
        got = np.ascontiguousarray(out.transpose(1, 0, 2))

    ref = reference(k, x, b)
    max_err = float(np.abs(got - ref).max())
    print(f"256x256x3 -> {OUT_ROWS}x{OUT_COLS}x{OUT_CHANNELS} stride 2  cycles={cycles}")
    print(f"cycles/output-row = {cycles / OUT_ROWS:.1f}")
    print(f"max abs err vs PyTorch reference: {max_err:.3e}")
    print("PASS" if max_err < 1e-2 else "FAIL")


if __name__ == "__main__":
    main()
