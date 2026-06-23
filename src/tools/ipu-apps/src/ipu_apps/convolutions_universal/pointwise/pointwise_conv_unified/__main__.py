"""CLI entry point for the unified pointwise convolution app."""

from __future__ import annotations

import argparse
from pathlib import Path

from ipu_apps.convolutions_universal.pointwise.pointwise_conv_unified import (
    PointwiseConvUnifiedApp,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run unified pointwise (1x1) convolution on the IPU emulator.",
    )
    parser.add_argument("--inst", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, default=None)
    parser.add_argument("--dtype", default="INT8")
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--in-channels", type=int, required=True)
    parser.add_argument("--out-channels", type=int, required=True)
    parser.add_argument("--max-cycles", type=int, default=10_000_000)
    args = parser.parse_args()

    app = PointwiseConvUnifiedApp(
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
