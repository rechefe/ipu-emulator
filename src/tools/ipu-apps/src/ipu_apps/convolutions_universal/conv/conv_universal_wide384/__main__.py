"""CLI entry point for the wide (W>=384) standard convolution app."""

from __future__ import annotations

import argparse
from pathlib import Path

from ipu_apps.convolutions_universal.conv.conv_universal_wide384 import (
    ConvUniversalWide384App,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run wide (W>=384) standard 3x3 convolution on the IPU emulator.",
    )
    parser.add_argument("--input", type=Path, required=True, help="Input image binary")
    parser.add_argument("--kernel", type=Path, required=True, help="Kernel binary")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output binary")
    parser.add_argument("--width", type=int, required=True, help="Spatial width (multiple of 128, >=384)")
    parser.add_argument("--rows", type=int, required=True, help="Spatial height")
    parser.add_argument("--in-channels", type=int, required=True, help="Input channels")
    parser.add_argument("--out-channels", type=int, required=True, help="Output channels (even)")
    parser.add_argument("--max-cycles", type=int, default=5_000_000, help="Max cycles")
    args = parser.parse_args()

    app = ConvUniversalWide384App(
        input_path=args.input,
        kernel_path=args.kernel,
        output_path=args.output,
        width=args.width,
        rows=args.rows,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
    )
    _, cycles = app.run(max_cycles=args.max_cycles)
    print(f"Completed in {cycles} cycles")


if __name__ == "__main__":
    main()
