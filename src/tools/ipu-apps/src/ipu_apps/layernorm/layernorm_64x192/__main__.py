"""Debug runner for layernorm_64x192.

Generates FP32 inputs with this kernel's own :mod:`gen_debug_data`, runs the
kernel, and prints the cycle count and RunStats. It does NOT check results --
``test/test_layernorm_64x192_wide.py`` is the reference for correctness.

Usage::

    LAYERNORM_64X192_INST_BIN=/tmp/layernorm_64x192.bin \
    uv run python -m ipu_apps.layernorm.layernorm_64x192
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.layernorm.layernorm_64x192 import LayerNorm64x192App
from ipu_apps.layernorm.layernorm_64x192.gen_debug_data import generate

_INST_BIN = Path(os.environ["LAYERNORM_64X192_INST_BIN"])


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="layernorm_64x192_"))
    kwargs = generate(work)

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    app = LayerNorm64x192App(
        inst_path=_INST_BIN,
        output_path=work / "output.bin",
        **kwargs,
    )
    state, cycles = app.run(max_cycles=5_000_000, state=state)
    print(f"Done in {cycles} cycles. Inputs/output under {work}")
    print(state.stats.format_summary())


if __name__ == "__main__":
    main()
