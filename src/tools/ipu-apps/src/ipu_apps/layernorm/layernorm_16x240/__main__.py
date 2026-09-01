"""Debug runner for layernorm_16x240.

Generates FP32 inputs with this kernel's own :mod:`gen_debug_data`, runs the
kernel, and prints the cycle count and RunStats. It does NOT check results --
``test/test_layernorm_16x240_wide.py`` is the reference for correctness.

Usage::

    LAYERNORM_16X240_INST_BIN=/tmp/layernorm_16x240.bin \
    uv run python -m ipu_apps.layernorm.layernorm_16x240
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.layernorm.layernorm_16x240 import LayerNorm16x240App
from ipu_apps.layernorm.layernorm_16x240.gen_debug_data import generate


def main() -> None:
    _INST_BIN = Path(os.environ["LAYERNORM_16X240_INST_BIN"])

    work = Path(tempfile.mkdtemp(prefix="layernorm_16x240_"))
    kwargs = generate(work)

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    app = LayerNorm16x240App(
        inst_path=_INST_BIN,
        output_path=work / "output.bin",
        **kwargs,
    )
    state, cycles = app.run(max_cycles=5_000_000, state=state)
    print(f"Done in {cycles} cycles. Inputs/output under {work}")
    print(state.stats.format_summary())


if __name__ == "__main__":
    main()
