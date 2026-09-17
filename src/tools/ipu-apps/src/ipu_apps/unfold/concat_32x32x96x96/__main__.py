"""Debug runner for concat_32x32x96x96.

Generates FP32 inputs with this kernel's own :mod:`gen_debug_data`, runs the
kernel, and prints the cycle count and RunStats. It does NOT check results.

Usage::

    CONCAT_32X32X96X96_INST_BIN=/tmp/concat_32x32x96x96.bin \
    uv run python -m ipu_apps.unfold.concat_32x32x96x96
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.unfold.concat_32x32x96x96 import Concat32x32x96x96App
from ipu_apps.unfold.concat_32x32x96x96.gen_debug_data import generate


def main() -> None:
    _INST_BIN = Path(os.environ["CONCAT_32X32X96X96_INST_BIN"])

    work = Path(tempfile.mkdtemp(prefix="concat_32x32x96x96_"))
    kwargs = generate(work)

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    app = Concat32x32x96x96App(
        inst_path=_INST_BIN,
        output_path=work / "output.bin",
        **kwargs,
    )
    state, cycles = app.run(max_cycles=5_000_000, state=state)
    print(f"Done in {cycles} cycles. Inputs/output under {work}")
    print(state.stats.format_summary())


if __name__ == "__main__":
    main()
