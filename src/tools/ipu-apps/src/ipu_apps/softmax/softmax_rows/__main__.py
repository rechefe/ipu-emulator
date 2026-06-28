"""Debug runner for the row-softmax app.

Assembles the .asm fresh, runs a random batch, and prints the cycle count and
max error against a numpy reference. Mirrors the in-process pattern used by the
conv apps (no pre-assembled binary required).

Usage::

    PYTHONPATH=... python -m ipu_apps.softmax.softmax_rows --rows 128
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.softmax.softmax_rows import (
    SoftmaxRowsApp,
    LANES,
    ROW_BYTES,
)

ASM_PATH = Path(__file__).resolve().parent / "softmax_rows.asm"


def reference_softmax(x: np.ndarray) -> np.ndarray:
    z = np.exp(x - x.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run row-softmax with a numpy cross-check")
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--scale", type=float, default=5.0, help="logit magnitude")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-cycles", type=int, default=2_000_000)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    x = (rng.randn(args.rows, LANES) * args.scale).astype(np.float32)
    ref = reference_softmax(x)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        inst_file = tmp / "softmax_rows.bin"
        input_file = tmp / "input.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(inst_file))
        input_file.write_bytes(x.tobytes())

        app = SoftmaxRowsApp(
            inst_path=inst_file,
            input_path=input_file,
            output_path=None,
            rows=args.rows,
        )
        state, cycles = app.run(max_cycles=args.max_cycles)
        raw = state.xmem.read_address(app.output_base, args.rows * ROW_BYTES)
        out = np.frombuffer(raw, dtype=np.float32).reshape(args.rows, LANES)

    max_err = float(np.abs(out - ref).max())
    row_sums = out.sum(axis=1)
    print(f"rows={args.rows} cycles={cycles} ({cycles / args.rows:.1f} cyc/row)")
    print(f"max abs err vs numpy softmax: {max_err:.3e}")
    print(f"row sums in [{row_sums.min():.6f}, {row_sums.max():.6f}]")
    print("PASS" if max_err < 1e-4 else "FAIL")


if __name__ == "__main__":
    main()
