"""Ask which softmax app handles a given shape.

Usage::

    PYTHONPATH=... python -m ipu_apps.softmax --axis rows --n 300 --rows 8
    PYTHONPATH=... python -m ipu_apps.softmax --axis columns --width 100 --rows 64
    PYTHONPATH=... python -m ipu_apps.softmax --shape 32,300 --dim 1
    PYTHONPATH=... python -m ipu_apps.softmax --catalog

``--shape``/``--dim`` mirror ``torch.softmax(x, dim)``: pass the tensor shape
and the dim you'd normalise over. Exits 0 if an app covers the shape, 1 if not.
"""

from __future__ import annotations

import argparse
import sys

from ipu_apps.softmax.registry import catalog, lookup, lookup_torch


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Which softmax app (if any) handles this shape?"
    )
    parser.add_argument("--catalog", action="store_true",
                        help="print the full coverage table and exit")
    parser.add_argument("--axis", choices=("rows", "columns"),
                        help="'rows' = softmax along each row; 'columns' = down each column")
    parser.add_argument("--rows", type=int, help="number of rows in the input")
    parser.add_argument("--n", type=int, help="elements per row (axis=rows)")
    parser.add_argument("--width", type=int, help="columns per row (axis=columns)")
    parser.add_argument("--shape",
                        help="torch-style tensor shape, e.g. '32,300' (use with --dim)")
    parser.add_argument("--dim", type=int, default=-1,
                        help="torch-style dim to softmax over (default: -1, the last)")
    args = parser.parse_args()

    if args.catalog:
        print(catalog())
        return 0

    try:
        if args.shape:
            dims = [int(p) for p in args.shape.replace("x", ",").split(",") if p.strip()]
            verdict = lookup_torch(dims, dim=args.dim)
        else:
            if not args.axis or args.rows is None:
                parser.error(
                    "--axis and --rows are required "
                    "(or use --shape/--dim, or --catalog)"
                )
            verdict = lookup(axis=args.axis, rows=args.rows, n=args.n, width=args.width)
    except ValueError as exc:
        parser.error(str(exc))

    print(verdict.describe())
    return 0 if verdict.supported else 1


if __name__ == "__main__":
    sys.exit(main())
