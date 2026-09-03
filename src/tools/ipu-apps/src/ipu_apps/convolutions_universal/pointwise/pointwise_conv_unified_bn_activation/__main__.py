"""CLI entry point for the pointwise conv + BN bias + ReLU app (FP32 wide-vector)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ipu_apps.convolutions_universal.pointwise.pointwise_conv_unified_bn_activation import (
    PointwiseConvUnifiedBnActivationApp,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run unified pointwise (1x1) convolution + folded BN bias + ReLU "
            "on the IPU emulator (FP32 wide-vector debug mode). "
            "input/kernel/bias/output are raw float32 files."
        ),
    )
    parser.add_argument("--inst", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument(
        "--bias",
        type=Path,
        default=None,
        help="Optional raw float32 bias file, one element per output channel.",
    )
    parser.add_argument("--output", "-o", type=Path, default=None)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--in-channels", type=int, required=True)
    parser.add_argument("--out-channels", type=int, required=True)
    parser.add_argument("--max-cycles", type=int, default=10_000_000)
    args = parser.parse_args()

    bias = None
    if args.bias is not None:
        bias = np.frombuffer(args.bias.read_bytes(), dtype=np.float32)

    app = PointwiseConvUnifiedBnActivationApp(
        inst_path=args.inst,
        input_path=args.input,
        kernel_path=args.kernel,
        bias=bias,
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
