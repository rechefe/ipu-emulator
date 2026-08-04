"""Debug runner for attn_v_bcast_48.

Generates FP32 inputs with this kernel's own :mod:`gen_debug_data`, runs the
kernel, and prints the cycle count and RunStats. It does NOT check results --
``test/test_attn_v_bcast_48_wide.py`` is the reference for correctness.

Usage::

    ATTN_V_BCAST_48_INST_BIN=/tmp/attn_v_bcast_48.bin \
    uv run python -m ipu_apps.attn_v_bcast_48
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.attn_v_bcast_48 import AttnVBcast48App
from ipu_apps.attn_v_bcast_48.gen_debug_data import generate

_INST_BIN = Path(os.environ["ATTN_V_BCAST_48_INST_BIN"])


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="attn_v_bcast_48_"))
    kwargs = generate(work)

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    app = AttnVBcast48App(
        inst_path=_INST_BIN,
        output_path=work / "output.bin",
        **kwargs,
    )
    state, cycles = app.run(max_cycles=20_000_000, state=state)
    print(f"Done in {cycles} cycles. Inputs/output under {work}")
    print(state.stats.format_summary())


if __name__ == "__main__":
    main()
