"""Debug runner for unfold_16x16x192.

Generates FP32 inputs with this kernel's own :mod:`gen_debug_data`, runs the
kernel, and prints the cycle count and RunStats. It does NOT check results --
``test/test_unfold_16x16x192_wide.py`` is the reference for correctness.

Usage::

    UNFOLD_16X16X192_INST_BIN=/tmp/unfold_16x16x192.bin \
    uv run python -m ipu_apps.unfold.unfold_16x16x192
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.unfold.unfold_16x16x192 import Unfold16x16x192App
from ipu_apps.unfold.unfold_16x16x192.gen_debug_data import generate


def main() -> None:
    _INST_BIN = Path(os.environ["UNFOLD_16X16X192_INST_BIN"])

    work = Path(tempfile.mkdtemp(prefix="unfold_16x16x192_"))
    kwargs = generate(work)

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    app = Unfold16x16x192App(
        inst_path=_INST_BIN,
        output_path=work / "output.bin",
        **kwargs,
    )
    state, cycles = app.run(max_cycles=20_000_000, state=state)
    print(f"Done in {cycles} cycles. Inputs/output under {work}")
    print(state.stats.format_summary())


if __name__ == "__main__":
    main()
