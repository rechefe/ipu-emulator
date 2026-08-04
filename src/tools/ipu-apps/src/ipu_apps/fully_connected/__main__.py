"""Debug runner for fully_connected.

Generates FP32 inputs with this kernel's own :mod:`gen_debug_data`, runs the
kernel, and prints the cycle count and RunStats. It does NOT check results --
``test/test_fully_connected_wide.py`` is the reference for correctness.

Usage::

    FC_INST_BIN=/tmp/fully_connected.bin \
    uv run python -m ipu_apps.fully_connected
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.fully_connected import FullyConnectedApp
from ipu_apps.fully_connected.gen_debug_data import generate

_INST_BIN = Path(os.environ["FC_INST_BIN"])


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="fully_connected_"))
    kwargs = generate(work)

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    app = FullyConnectedApp(
        inst_path=_INST_BIN,
        output_path=work / "output.bin",
        **kwargs,
    )
    state, cycles = app.run(max_cycles=2_000_000, state=state)
    print(f"Done in {cycles} cycles. Inputs/output under {work}")
    print(state.stats.format_summary())


if __name__ == "__main__":
    main()
