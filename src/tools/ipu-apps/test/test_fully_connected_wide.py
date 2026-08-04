"""Wide-vector FP32 end-to-end test for the fully-connected kernel.

Runs the REAL kernel binary against a numpy reference in wide-vector debug
mode. No checked-in golden: FP32 inputs are generated here and the reference
out = in @ W.T is computed directly.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.fully_connected import (
    FullyConnectedApp, SAMPLES_NUM, INPUT_NEURONS, OUTPUT_NEURONS,
)

_INST_BIN = Path(os.environ["FC_INST_BIN"])


def test_fully_connected_wide_fp32(tmp_path: Path) -> None:
    rng = np.random.RandomState(0xFC0)

    # in: [SAMPLES_NUM, INPUT_NEURONS], W: [OUTPUT_NEURONS, INPUT_NEURONS]
    inputs = rng.uniform(-1.0, 1.0, size=(SAMPLES_NUM, INPUT_NEURONS)).astype(np.float32)
    weights = rng.uniform(-1.0, 1.0, size=(OUTPUT_NEURONS, INPUT_NEURONS)).astype(np.float32)

    inputs_path = tmp_path / "inputs_fp32.bin"
    weights_path = tmp_path / "weights_fp32.bin"
    inputs_path.write_bytes(inputs.tobytes())
    weights_path.write_bytes(weights.tobytes())
    output_path = tmp_path / "output.bin"

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    app = FullyConnectedApp(
        inst_path=_INST_BIN,
        inputs_path=inputs_path,
        weights_path=weights_path,
        output_path=output_path,
    )
    state, cycles = app.run(max_cycles=2_000_000, state=state)
    assert cycles > 0

    expected = inputs @ weights.T          # [SAMPLES_NUM, OUTPUT_NEURONS]

    raw = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    assert raw.size == SAMPLES_NUM * OUTPUT_NEURONS, (
        f"output has {raw.size} floats, expected {SAMPLES_NUM * OUTPUT_NEURONS}"
    )
    got = raw.reshape(SAMPLES_NUM, OUTPUT_NEURONS)

    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-3)
