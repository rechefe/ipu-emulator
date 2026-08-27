"""Debug runner for the packed sub-128 column-softmax app.

Assembles the .asm fresh, runs a random batch, and prints the cycle count and
max error against a numpy column-softmax reference (softmax DOWN each column).

Usage::

    PYTHONPATH=... python -m ipu_apps.softmax.softmax_columns_packed --rows 100 --width 16
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.softmax.softmax_columns_packed import SoftmaxColumnsPackedApp

ASM_PATH = Path(__file__).resolve().parent / "softmax_columns_packed.asm"


def reference_softmax(x: np.ndarray) -> np.ndarray:
    z = np.exp(x - x.max(axis=0, keepdims=True))
    return z / z.sum(axis=0, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run packed sub-128 column-softmax with a numpy cross-check"
    )
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--width", type=int, default=16, help="real elements per row (1..64)")
    parser.add_argument("--scale", type=float, default=3.0, help="logit magnitude")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-cycles", type=int, default=8_000_000)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    x = (rng.randn(args.rows, args.width) * args.scale).astype(np.float32)
    ref = reference_softmax(x)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        inst_file = tmp / "softmax_columns_packed.bin"
        input_file = tmp / "input.bin"
        output_file = tmp / "output.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(inst_file))
        input_file.write_bytes(x.tobytes())

        app = SoftmaxColumnsPackedApp(
            inst_path=inst_file,
            input_path=input_file,
            output_path=output_file,
            rows=args.rows,
            width=args.width,
        )
        _, cycles = app.run(max_cycles=args.max_cycles)
        out = np.frombuffer(output_file.read_bytes(), dtype=np.float32).reshape(args.rows, args.width)

    max_err = float(np.abs(out - ref).max())
    col_sums = out.sum(axis=0)
    print(f"rows={args.rows} width={args.width} (rpv={app.rows_per_vec}, "
          f"{app.num_vectors} vectors) cycles={cycles}")
    print(f"max abs err vs numpy column-softmax: {max_err:.3e}")
    print(f"column sums in [{col_sums.min():.6f}, {col_sums.max():.6f}]")
    print("PASS" if max_err < 1e-4 else "FAIL")


if __name__ == "__main__":
    main()
