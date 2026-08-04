"""Debug runner for qk_scores_16x60.

Generates FP32 inputs with this kernel's own :mod:`gen_debug_data`, runs the
kernel, and prints the cycle count and RunStats. It does NOT check results --
``test/test_qk_scores_16x60_wide.py`` is the reference for correctness.

Usage::

    QK_SCORES_16X60_INST_BIN=/tmp/qk_scores_16x60.bin \
    uv run python -m ipu_apps.qk_scores_16x60
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.qk_scores_16x60 import QkScores16x60App
from ipu_apps.qk_scores_16x60.gen_debug_data import generate

_INST_BIN = Path(os.environ["QK_SCORES_16X60_INST_BIN"])


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="qk_scores_16x60_"))
    kwargs = generate(work)

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    app = QkScores16x60App(
        inst_path=_INST_BIN,
        output_path=work / "output.bin",
        **kwargs,
    )
    state, cycles = app.run(max_cycles=20_000_000, state=state)
    print(f"Done in {cycles} cycles. Inputs/output under {work}")
    print(state.stats.format_summary())


if __name__ == "__main__":
    main()
