"""CLI entry point for the depthwise stride-2 (cols=128) conv app (FP32 wide-vector)."""

from __future__ import annotations

import argparse
from pathlib import Path

from ipu_apps.convolutions_universal.depthwise.depthwise_conv_stride2_128 import (
    DepthwiseConvStride2_128App,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run depthwise 3x3 stride-2 conv (cols=128, no bias/ReLU) "
        "on the IPU emulator (FP32 wide-vector debug mode). "
        "input/kernel/output are raw float32 files.",
    )
    parser.add_argument("--input", type=Path, required=True, help="Input image binary")
    parser.add_argument("--kernel", type=Path, required=True, help="Kernel binary (channels*9*4 bytes)")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output binary")
    parser.add_argument("--rows", type=int, required=True, help="Spatial height (multiple of 4)")
    parser.add_argument("--channels", type=int, required=True, help="Number of channels")
    parser.add_argument("--max-cycles", type=int, default=2_000_000, help="Max cycles")
    args = parser.parse_args()

    app = DepthwiseConvStride2_128App(
        input_path=args.input,
        kernel_path=args.kernel,
        output_path=args.output,
        rows=args.rows,
        channels=args.channels,
    )
    _, cycles = app.run(max_cycles=args.max_cycles)
    print(f"Completed in {cycles} cycles")


if __name__ == "__main__":
    main()
