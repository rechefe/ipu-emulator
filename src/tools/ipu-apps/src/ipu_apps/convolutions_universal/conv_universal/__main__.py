"""CLI entry point for the universal standard convolution app."""

from __future__ import annotations

import argparse
from pathlib import Path

from ipu_apps.convolutions_universal.conv_universal import ConvUniversalApp


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run universal standard 3x3 convolution on the IPU emulator.",
    )
    parser.add_argument("--inst", type=Path, required=True, help="Assembled binary")
    parser.add_argument("--input", type=Path, required=True, help="Input image binary")
    parser.add_argument("--kernel", type=Path, required=True, help="Kernel binary")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output binary")
    parser.add_argument("--dtype", default="INT8", help="Data type (INT8, FP8_E4M3, FP8_E5M2)")
    parser.add_argument("--rows", type=int, required=True, help="Spatial height")
    parser.add_argument("--cols", type=int, required=True, help="Spatial width")
    parser.add_argument("--in-channels", type=int, required=True, help="Input channels")
    parser.add_argument("--out-channels", type=int, required=True, help="Output channels")
    parser.add_argument("--max-cycles", type=int, default=50_000_000, help="Max cycles")
    args = parser.parse_args()

    app = ConvUniversalApp(
        inst_path=args.inst,
        input_path=args.input,
        kernel_path=args.kernel,
        output_path=args.output,
        dtype=args.dtype,
        rows=args.rows,
        cols=args.cols,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
    )
    _, cycles = app.run(max_cycles=args.max_cycles)
    print(f"Completed in {cycles} cycles")


if __name__ == "__main__":
    main()
