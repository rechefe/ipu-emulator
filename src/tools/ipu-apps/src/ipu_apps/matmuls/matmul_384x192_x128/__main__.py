"""Debug runner for matmul_384x192_x128.

Generates FP32 inputs with this kernel's own :mod:`gen_debug_data`, runs the
kernel, and prints the cycle count and RunStats. It does NOT check results --
``test/test_matmul_384x192_x128_wide.py`` is the reference for correctness.

Usage::

    MATMUL_384X192_X128_INST_BIN=/tmp/matmul_384x192_x128.bin \
    uv run python -m ipu_apps.matmuls.matmul_384x192_x128
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.matmuls.matmul_384x192_x128 import MatMul384x192x128App
from ipu_apps.matmuls.matmul_384x192_x128.gen_debug_data import generate

_INST_BIN = Path(os.environ["MATMUL_384X192_X128_INST_BIN"])


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="matmul_384x192_x128_"))
    kwargs = generate(work)

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    app = MatMul384x192x128App(
        inst_path=_INST_BIN,
        output_path=work / "output.bin",
        **kwargs,
    )
    state, cycles = app.run(max_cycles=20_000_000, state=state)
    print(f"Done in {cycles} cycles. Inputs/output under {work}")
    print(state.stats.format_summary())


if __name__ == "__main__":
    main()
