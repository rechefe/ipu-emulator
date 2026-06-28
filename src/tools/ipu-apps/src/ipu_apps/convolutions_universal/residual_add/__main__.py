"""Debug runner for the wide-vector FP32 residual-add app.

Assembles the .asm fresh, runs two random FP32 tensors, and prints the cycle
count and max error against a numpy reference. Mirrors the in-process pattern
used by the softmax wide-vector apps (no pre-assembled binary required).

Usage::

    PYTHONPATH=... python -m ipu_apps.convolutions_universal.residual_add --channels 160
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.convolutions_universal.residual_add import (
    ResidualAddApp,
    LANES_PER_CHUNK,
    WIDE_CHUNK_BYTES,
)

ASM_PATH = Path(__file__).resolve().parent / "residual_add.asm"


def _pack(values: np.ndarray) -> bytes:
    return values.astype("<f4").tobytes()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run residual add with a numpy cross-check")
    parser.add_argument("--channels", type=int, default=160)
    parser.add_argument("--scale", type=float, default=4.0, help="input magnitude")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-cycles", type=int, default=2_000_000)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    shape = (args.channels, LANES_PER_CHUNK)
    a = (rng.randn(*shape) * args.scale).astype(np.float32)
    b = (rng.randn(*shape) * args.scale).astype(np.float32)
    ref = (a + b).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        inst_file = tmp / "residual_add.bin"
        a_file = tmp / "input_a.bin"
        b_file = tmp / "input_b.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(inst_file))
        a_file.write_bytes(_pack(a))
        b_file.write_bytes(_pack(b))

        app = ResidualAddApp(
            inst_path=inst_file,
            input_a_path=a_file,
            input_b_path=b_file,
            output_path=None,
            num_channels=args.channels,
        )
        state, cycles = app.run(max_cycles=args.max_cycles)
        raw = state.xmem.read_address(app.output_base, args.channels * WIDE_CHUNK_BYTES)
        out = np.frombuffer(raw, dtype=np.float32).reshape(args.channels, LANES_PER_CHUNK)

    max_err = float(np.abs(out - ref).max())
    print(f"channels={args.channels} cycles={cycles} ({cycles / args.channels:.1f} cyc/ch)")
    print(f"max abs err vs numpy a+b: {max_err:.3e}")
    print("PASS" if max_err == 0.0 else "FAIL")


if __name__ == "__main__":
    main()
