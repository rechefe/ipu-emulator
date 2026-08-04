"""Debug runner for unfold_32x32x144.

Generates FP32 inputs with this kernel's own :mod:`gen_debug_data`, runs the
kernel, and prints the cycle count and RunStats. It does NOT check results --
``test/test_unfold_32x32x144_wide.py`` is the reference for correctness.

Usage::

    UNFOLD_32X32X144_INST_BIN=/tmp/unfold_32x32x144.bin \
    uv run python -m ipu_apps.unfold_32x32x144
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.unfold_32x32x144 import Unfold32x32x144App
from ipu_apps.unfold_32x32x144.gen_debug_data import generate

_INST_BIN = Path(os.environ["UNFOLD_32X32X144_INST_BIN"])


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="unfold_32x32x144_"))
    kwargs = generate(work)

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    app = Unfold32x32x144App(
        inst_path=_INST_BIN,
        output_path=work / "output.bin",
        **kwargs,
    )
    state, cycles = app.run(max_cycles=20_000_000, state=state)
    print(f"Done in {cycles} cycles. Inputs/output under {work}")
    print(state.stats.format_summary())


if __name__ == "__main__":
    main()
