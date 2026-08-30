"""Debug runner for the long-row softmax app (N>128, N%128 != 0).

Assembles the .asm fresh, runs a random batch, and prints the cycle count and
max error against a numpy reference.

Usage::

    PYTHONPATH=... python -m ipu_apps.softmax.softmax_rows_long --rows 8 --n 300
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.softmax.softmax_rows_long import SoftmaxRowsLongApp

ASM_PATH = Path(__file__).resolve().parent / "softmax_rows_long.asm"


def reference_softmax(x: np.ndarray) -> np.ndarray:
    z = np.exp(x - x.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run long-row softmax (N>128, N%128 != 0) with a numpy cross-check"
    )
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--n", type=int, default=300, help="elements per row (>128, %%128 != 0)")
    parser.add_argument("--scale", type=float, default=5.0, help="logit magnitude")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-cycles", type=int, default=8_000_000)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    x = (rng.randn(args.rows, args.n) * args.scale).astype(np.float32)
    ref = reference_softmax(x)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        inst_file = tmp / "softmax_rows_long.bin"
        input_file = tmp / "input.bin"
        output_file = tmp / "output.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(inst_file))
        input_file.write_bytes(x.tobytes())

        app = SoftmaxRowsLongApp(
            inst_path=inst_file,
            input_path=input_file,
            output_path=output_file,
            rows=args.rows,
            n=args.n,
        )
        _, cycles = app.run(max_cycles=args.max_cycles)
        out = np.frombuffer(output_file.read_bytes(), dtype=np.float32).reshape(args.rows, args.n)

    max_err = float(np.abs(out - ref).max())
    row_sums = out.sum(axis=1)
    print(f"rows={args.rows} n={args.n} cycles={cycles} ({cycles / args.rows:.1f} cyc/row)")
    print(f"max abs err vs numpy softmax: {max_err:.3e}")
    print(f"row sums in [{row_sums.min():.6f}, {row_sums.max():.6f}]")
    print("PASS" if max_err < 1e-4 else "FAIL")


if __name__ == "__main__":
    main()
