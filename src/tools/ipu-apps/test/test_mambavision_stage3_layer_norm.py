"""Regression test for MambaVision Stage 3 Layer Norm (wide-vector FP32).

Compares the emulator's output against a Python golden reference using a
numeric tolerance rather than byte-exact equality: unlike the INT8 kernels
(deterministic integer arithmetic), this kernel does real FP32 math, and the
golden reference's summation order (a single running sum over all 320
channels) only approximates the kernel's (per-tile sums combined at the end)
-- floating point addition is not associative, so small last-digit
differences are expected. Measured worst-case absolute error across all
196*320 outputs is ~0.011; the 0.05 tolerance below leaves comfortable
margin without masking a real correctness bug.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

from ipu_apps.mambavision_stage3_layer_norm import (
    CHANNELS,
    TOKENS,
    MambavisionStage3LayerNormApp,
)

_INST_BIN = Path(os.environ["LAYER_NORM_INST_BIN"])
_DATA_DIR = Path(os.environ["LAYER_NORM_DATA_DIR"])

_ABS_TOLERANCE = 0.05


def _read_f32(path: Path, count: int) -> list[float]:
    raw = path.read_bytes()
    return list(struct.unpack_from(f"<{count}f", raw, 0))


def test_mambavision_stage3_layer_norm(tmp_path: Path) -> None:
    data_dir = _DATA_DIR / "fp32"
    output = tmp_path / "output_fp32.bin"

    app = MambavisionStage3LayerNormApp(
        inst_path=_INST_BIN,
        inputs_path=data_dir / "inputs_fp32.bin",
        gamma_path=data_dir / "gamma_fp32.bin",
        beta_path=data_dir / "beta_fp32.bin",
        const_path=data_dir / "const_fp32.bin",
        output_path=output,
    )

    _state, cycles = app.run(max_cycles=5_000_000)
    assert cycles > 0

    golden_path = data_dir / "out_fp32_layer_norm.bin"
    assert golden_path.exists(), f"Missing golden file: {golden_path}"

    total = TOKENS * CHANNELS
    actual = _read_f32(output, total)
    golden = _read_f32(golden_path, total)

    max_abs_err = 0.0
    worst_idx = -1
    for i, (a, g) in enumerate(zip(actual, golden)):
        err = abs(a - g)
        if err > max_abs_err:
            max_abs_err = err
            worst_idx = i

    assert max_abs_err < _ABS_TOLERANCE, (
        f"Worst mismatch at flat index {worst_idx} "
        f"(token={worst_idx // CHANNELS}, channel={worst_idx % CHANNELS}): "
        f"actual={actual[worst_idx]!r} golden={golden[worst_idx]!r} "
        f"abs_err={max_abs_err!r}"
    )
