"""Runnable demo for bootcamp drill 1: residual_add.

Assembles residual_add.asm fresh, runs it on random INT8 tensors, and prints
the cycle count plus a pass/fail check against a numpy reference.

Usage::

    PYTHONPATH=... python -m ipu_apps.bootcamp.residual_add --vectors 8
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.bootcamp.residual_add import ResidualAddApp, LANES, ROW_BYTES_OUT

ASM_PATH = Path(__file__).resolve().parent / "residual_add.asm"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bootcamp residual_add with a numpy cross-check")
    parser.add_argument("--vectors", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-cycles", type=int, default=100_000)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    shape = (args.vectors, LANES)
    # Keep values small enough that INT8 + INT8 has an obviously-correct
    # INT32 sum with no surprises -- this drill is about the ISA plumbing,
    # not about overflow edge cases.
    a = rng.randint(-64, 64, size=shape).astype(np.int8)
    b = rng.randint(-64, 64, size=shape).astype(np.int8)
    ref = a.astype(np.int32) + b.astype(np.int32)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        inst_file = tmp / "residual_add.bin"
        a_file = tmp / "input_a.bin"
        b_file = tmp / "input_b.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(inst_file))
        a_file.write_bytes(a.tobytes())
        b_file.write_bytes(b.tobytes())

        app = ResidualAddApp(
            inst_path=inst_file,
            input_a_path=a_file,
            input_b_path=b_file,
            output_path=None,
            num_vectors=args.vectors,
        )
        state, cycles = app.run(max_cycles=args.max_cycles)
        raw = state.xmem.read_address(app.output_base, args.vectors * ROW_BYTES_OUT)
        out = np.frombuffer(raw, dtype="<i4").reshape(args.vectors, LANES)

    max_err = int(np.abs(out.astype(np.int64) - ref.astype(np.int64)).max())
    print(f"vectors={args.vectors} cycles={cycles} ({cycles / args.vectors:.1f} cyc/vec)")
    print(f"max abs err vs numpy a+b: {max_err}")
    print("PASS" if max_err == 0 else "FAIL")


if __name__ == "__main__":
    main()
