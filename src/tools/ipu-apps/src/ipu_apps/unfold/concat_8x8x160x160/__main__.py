"""Debug runner for concat_8x8x160x160.

Generates FP32 inputs with this kernel's own :mod:`gen_debug_data`, runs the
kernel, and prints the cycle count and RunStats. It does NOT check results.

Usage::

    CONCAT_8X8X160X160_INST_BIN=/tmp/concat_8x8x160x160.bin \
    uv run python -m ipu_apps.unfold.concat_8x8x160x160
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.unfold.concat_8x8x160x160 import Concat8x8x160x160App
from ipu_apps.unfold.concat_8x8x160x160.gen_debug_data import generate


def main() -> None:
    _INST_BIN = Path(os.environ["CONCAT_8X8X160X160_INST_BIN"])

    work = Path(tempfile.mkdtemp(prefix="concat_8x8x160x160_"))
    kwargs = generate(work)

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    app = Concat8x8x160x160App(
        inst_path=_INST_BIN,
        output_path=work / "output.bin",
        **kwargs,
    )
    state, cycles = app.run(max_cycles=5_000_000, state=state)
    print(f"Done in {cycles} cycles. Inputs/output under {work}")
    print(state.stats.format_summary())


if __name__ == "__main__":
    main()
