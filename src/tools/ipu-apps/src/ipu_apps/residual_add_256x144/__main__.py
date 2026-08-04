"""Debug runner for residual_add_256x144.

Generates FP32 inputs with this kernel's own :mod:`gen_debug_data`, runs the
kernel, and prints the cycle count and RunStats. It does NOT check results --
``test/test_residual_add_256x144_wide.py`` is the reference for correctness.

Usage::

    RESIDUAL_ADD_256X144_INST_BIN=/tmp/residual_add_256x144.bin \
    uv run python -m ipu_apps.residual_add_256x144
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.residual_add_256x144 import ResidualAdd256x144App
from ipu_apps.residual_add_256x144.gen_debug_data import generate

_INST_BIN = Path(os.environ["RESIDUAL_ADD_256X144_INST_BIN"])


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="residual_add_256x144_"))
    kwargs = generate(work)

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    app = ResidualAdd256x144App(
        inst_path=_INST_BIN,
        output_path=work / "output.bin",
        **kwargs,
    )
    state, cycles = app.run(max_cycles=5_000_000, state=state)
    print(f"Done in {cycles} cycles. Inputs/output under {work}")
    print(state.stats.format_summary())


if __name__ == "__main__":
    main()
