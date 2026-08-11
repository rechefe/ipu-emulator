"""Debug runner for proj_outproj_144_p4.

Generates FP32 inputs for all 4 streams plus shared weights, runs the
kernel, and prints the cycle count and RunStats. It does NOT check
results -- ``test/test_proj_outproj_144_p4_wide.py`` is the reference for
correctness.

Usage::

    PROJ_OUTPROJ_144_P4_INST_BIN=/tmp/proj_outproj_144_p4.bin \
    uv run python -m ipu_apps.proj_outproj_144_p4
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.proj_outproj_144_p4 import (
    ProjOutproj144P4App, K, N_OUT, N_TG, N_TOK, N_STREAM,
)

_INST_BIN = Path(os.environ["PROJ_OUTPROJ_144_P4_INST_BIN"])
_SEED = 0xC0FFEE


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="proj_outproj_144_p4_"))
    rng = np.random.RandomState(_SEED)

    input_paths = []
    for p in range(N_STREAM):
        D = rng.uniform(-1.0, 1.0, size=(N_TG, K, N_TOK)).astype(np.float32)
        path = work / f"input_fp32_p{p}.bin"
        path.write_bytes(D.tobytes())
        input_paths.append(path)

    W = rng.uniform(-1.0, 1.0, size=(N_OUT, K)).astype(np.float32)
    weights_path = work / "weights_fp32.bin"
    weights_path.write_bytes(W.tobytes())

    output_paths = [work / f"output_p{p}.bin" for p in range(N_STREAM)]

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    app = ProjOutproj144P4App(
        inst_path=_INST_BIN,
        input_paths=input_paths,
        weights_path=weights_path,
        output_paths=output_paths,
    )
    state, cycles = app.run(max_cycles=20_000_000, state=state)
    print(f"Done in {cycles} cycles. Inputs/output under {work}")
    print(state.stats.format_summary())


if __name__ == "__main__":
    main()
