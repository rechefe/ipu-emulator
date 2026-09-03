"""Ask which conv2d kernel handles a given shape.

Usage::

    PYTHONPATH=... python -m ipu_apps.convolutions_universal --in-channels 32 --out-channels 32 \\
        --kernel-size 3 --stride 1 --padding 1 --groups 32 --height 64 --width 64
    PYTHONPATH=... python -m ipu_apps.convolutions_universal --catalog

Exits 0 if a kernel covers the shape, 1 if not.
"""

from __future__ import annotations

import argparse
import sys

from ipu_apps.convolutions_universal.registry import catalog, lookup


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Which conv2d kernel (if any) handles this shape?"
    )
    parser.add_argument("--catalog", action="store_true",
                        help="print the full coverage table and exit")
    parser.add_argument("--in-channels", type=int)
    parser.add_argument("--out-channels", type=int)
    parser.add_argument("--kernel-size", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--padding", type=int, default=0)
    parser.add_argument("--dilation", type=int, default=1)
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--bias", action="store_true", help="query has_bias=True")
    parser.add_argument("--relu", action="store_true", help="query apply_relu=True")
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    args = parser.parse_args()

    if args.catalog:
        print(catalog())
        return 0

    required = ("in_channels", "out_channels", "kernel_size", "height", "width")
    missing = [
        f"--{name.replace('_', '-')}" for name in required
        if getattr(args, name) is None
    ]
    if missing:
        parser.error(f"missing required argument(s): {', '.join(missing)} (or use --catalog)")

    verdict = lookup(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        kernel_size=args.kernel_size,
        stride=args.stride,
        padding=args.padding,
        dilation=args.dilation,
        groups=args.groups,
        has_bias=args.bias,
        apply_relu=args.relu,
        height=args.height,
        width=args.width,
    )
    print(verdict.describe())
    return 0 if verdict.supported else 1


if __name__ == "__main__":
    sys.exit(main())
