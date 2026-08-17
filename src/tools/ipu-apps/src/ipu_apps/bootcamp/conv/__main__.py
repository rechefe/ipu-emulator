"""Runnable demo for bootcamp drill 2: conv (1D single-channel 3-tap conv).

Assembles conv.asm fresh, runs it on random INT8 rows, and prints the cycle
count plus a pass/fail check against a numpy reference (valid conv, no
padding, discarding edge lanes 0 and 127 -- see conv.asm and __init__.py).

Usage::

    PYTHONPATH=... python -m ipu_apps.bootcamp.conv --rows 4
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.bootcamp.conv import ConvApp, LANES, ROW_BYTES_OUT, VALID_LO, VALID_HI

ASM_PATH = Path(__file__).resolve().parent / "conv.asm"


def _numpy_reference(rows: np.ndarray, weights: tuple[int, int, int]) -> np.ndarray:
    """Valid 1D conv, positions 1..126, matching conv.asm's shift-and-sum."""
    w_left, w_center, w_right = weights
    rows64 = rows.astype(np.int64)
    left = rows64[:, VALID_LO - 1 : VALID_HI - 1]
    center = rows64[:, VALID_LO:VALID_HI]
    right = rows64[:, VALID_LO + 1 : VALID_HI + 1]
    return w_left * left + w_center * center + w_right * right


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bootcamp conv with a numpy cross-check")
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-cycles", type=int, default=100_000)
    args = parser.parse_args()

    weights = (1, 2, -1)
    rng = np.random.RandomState(args.seed)
    data = rng.randint(-32, 32, size=(args.rows, LANES)).astype(np.int8)
    ref = _numpy_reference(data, weights)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        inst_file = tmp / "conv.bin"
        input_file = tmp / "input.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(inst_file))
        input_file.write_bytes(data.tobytes())

        app = ConvApp(
            inst_path=inst_file,
            input_path=input_file,
            output_path=None,
            num_rows=args.rows,
            weights=weights,
        )
        state, cycles = app.run(max_cycles=args.max_cycles)
        raw = state.xmem.read_address(app.output_base, args.rows * ROW_BYTES_OUT)
        out = np.frombuffer(raw, dtype="<i4").reshape(args.rows, LANES)
        out_valid = out[:, VALID_LO:VALID_HI]

    max_err = int(np.abs(out_valid.astype(np.int64) - ref).max())
    print(f"rows={args.rows} weights={weights} cycles={cycles} ({cycles / args.rows:.1f} cyc/row)")
    print(f"max abs err vs numpy valid-conv: {max_err}")
    print("PASS" if max_err == 0 else "FAIL")


if __name__ == "__main__":
    main()
