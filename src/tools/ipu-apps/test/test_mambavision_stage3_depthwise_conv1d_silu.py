"""Regression test for MambaVision Stage 3 depthwise Conv1D + SiLU (FP32).

Byte-exact, not tolerance-based: inputs/weights are integer-valued (stored as
float32, exactly representable well under the 2^24 exact-integer range), so
the 3-tap conv accumulation is bit-exact regardless of summation order --
unlike Layer Norm, there's no genuinely-fractional intermediate being summed
here. The golden reference's SiLU implementation mirrors
ipu_common/activations.py's _sigmoid formula exactly (same branching, same
math.exp calls) so it matches the emulator bit-for-bit, not just numerically.
"""

from __future__ import annotations

import os
from pathlib import Path

from ipu_apps.mambavision_stage3_depthwise_conv1d_silu import (
    CHANNELS,
    TOKENS,
    MambavisionStage3DepthwiseConv1dSiluApp,
)

_INST_BIN = Path(os.environ["CONV1D_SILU_INST_BIN"])
_DATA_DIR = Path(os.environ["CONV1D_SILU_DATA_DIR"])


def test_mambavision_stage3_depthwise_conv1d_silu(tmp_path: Path) -> None:
    data_dir = _DATA_DIR / "fp32"
    output = tmp_path / "output_fp32.bin"

    app = MambavisionStage3DepthwiseConv1dSiluApp(
        inst_path=_INST_BIN,
        inputs_padded_path=data_dir / "inputs_padded_fp32.bin",
        tap_minus1_path=data_dir / "tap_minus1_fp32.bin",
        tap_zero_path=data_dir / "tap_zero_fp32.bin",
        tap_plus1_path=data_dir / "tap_plus1_fp32.bin",
        output_path=output,
    )

    _state, cycles = app.run(max_cycles=5_000_000)
    assert cycles > 0

    golden_path = data_dir / "out_fp32_conv1d_silu.bin"
    assert golden_path.exists(), f"Missing golden file: {golden_path}"

    expected_size = TOKENS * CHANNELS * 4
    actual_bytes = output.read_bytes()
    golden_bytes = golden_path.read_bytes()
    assert len(actual_bytes) == expected_size
    assert actual_bytes == golden_bytes
