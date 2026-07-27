"""Debug runner for the first-layer conv app (256x256x3 -> 128x128x16, stride 2).

Assembles the .asm fresh, runs a random INT8 batch, and prints the cycle count and
max error against a numpy pad=1 stride-2 conv reference (with the app's
intermediate-clamp + ReLU datapath).

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
    OUT_ROW_GROUP,
    IN_ROWS,
    IN_CHANNELS,
    IN_COLS,
)
from ipu_apps.convolutions_universal.conv.conv_first_layer.test_conv_first_layer import (
    reference,
)

ASM_PATH = ConvFirstLayerApp.ASM_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Run first-layer conv with a numpy cross-check")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-cycles", type=int, default=500_000_000)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    x = rng.randint(-3, 4, size=IN_ROWS * IN_CHANNELS * IN_COLS, dtype=np.int8)
    k = rng.randint(-3, 4, size=(OUT_CHANNELS, IN_CHANNELS, 3, 3)).astype(np.int8)
    b = rng.randint(-5, 6, size=OUT_CHANNELS).astype(np.int8)
    input_bytes = x.view(np.uint8).tobytes()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        inst = tmp / "conv_first_layer.bin"
        infile = tmp / "input.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(inst))
        infile.write_bytes(input_bytes)

        app = ConvFirstLayerApp(
            inst_path=inst, input_path=infile, kernel=k, bias=b, output_path=None
        )
        state, cycles = app.run(max_cycles=args.max_cycles)
        total = OUT_ROWS * OUT_ROW_GROUP
        got = np.frombuffer(bytes(state.xmem.read_address(OUTPUT_BASE_ADDR, total)), dtype=np.int8)

    ref = np.frombuffer(reference(input_bytes, k, b), dtype=np.int8)
    max_err = int(np.abs(got.astype(np.int32) - ref.astype(np.int32)).max())
    print(f"256x256x3 -> {OUT_ROWS}x{OUT_COLS}x{OUT_CHANNELS} stride 2  cycles={cycles}")
    print(f"cycles/output-row = {cycles / OUT_ROWS:.1f}")
    print(f"max abs err vs numpy reference: {max_err}")
    print("PASS" if max_err == 0 else "FAIL")


if __name__ == "__main__":
    main()
