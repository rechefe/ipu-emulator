"""CLI entry point for the universal standard convolution app (FP32 wide-vector)."""

from __future__ import annotations

import argparse
from pathlib import Path

from ipu_apps.convolutions_universal.conv.conv_universal import ConvUniversalApp


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run universal standard 3x3 convolution on the IPU emulator "
        "(FP32 wide-vector debug mode). input/kernel/output are raw float32 files.",
    )
    parser.add_argument("--inst", type=Path, required=True, help="Assembled binary")
    parser.add_argument("--input", type=Path, required=True, help="Input image binary")
    parser.add_argument("--kernel", type=Path, required=True, help="Kernel binary")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output binary")
    parser.add_argument("--height", type=int, required=True, help="Spatial height")
    parser.add_argument("--width", type=int, required=True, help="Spatial width")
    parser.add_argument("--in-channels", type=int, required=True, help="Input channels")
    parser.add_argument("--out-channels", type=int, required=True, help="Output channels")
    parser.add_argument("--max-cycles", type=int, default=50_000_000, help="Max cycles")
    args = parser.parse_args()

    app = ConvUniversalApp(
        inst_path=args.inst,
        input_path=args.input,
        kernel_path=args.kernel,
        output_path=args.output,
        height=args.height,
        width=args.width,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
    )
    _, cycles = app.run(max_cycles=args.max_cycles)
    print(f"Completed in {cycles} cycles")


if __name__ == "__main__":
    main()
