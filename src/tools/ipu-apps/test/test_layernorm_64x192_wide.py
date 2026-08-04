"""Wide-vector FP32 end-to-end test for layernorm_64x192 (Layer 4).

Runs the REAL kernel binary against an inline numpy reference in wide-vector
debug mode. No checked-in golden and no data directory: FP32 inputs are
generated here and the expected result is computed directly.

Results are read back through the kernel's own store path
(ACTIVATE.QUANTIZE -> STR_POST_AAQ_REG -> dump_xmem_to_binary), not out of
R_ACC.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.layernorm_64x192 import (
    LayerNorm64x192App, N_CH, N_TOK, LANES, ROW_BYTES,
)

_INST_BIN = Path(os.environ["LAYERNORM_64X192_INST_BIN"])


def _reference_layernorm(
    x: np.ndarray, gamma: np.ndarray, beta: np.ndarray
) -> np.ndarray:
    """output[ch, i] = γ[ch] × (x[ch,i] − μ[i]) / σ[i] + β[ch].

    μ/σ reduce over the CHANNEL axis, independently per token. No epsilon --
    this matches the rsqrt activation, which handles stability internally.

    x: [N_CH, N_TOK], gamma/beta: [N_CH]  ->  [N_CH, N_TOK]
    """
    mean = x.mean(axis=0)                                   # [N_TOK]
    centered = x - mean                                     # [N_CH, N_TOK]
    var = (centered ** 2).mean(axis=0)                      # [N_TOK]
    inv_std = np.where(var > 0.0, 1.0 / np.sqrt(var), 0.0)
    normalized = centered * inv_std
    return (gamma[:, None] * normalized + beta[:, None]).astype(np.float32)


def test_layernorm_64x192_wide_fp32(tmp_path: Path) -> None:
    rng = np.random.RandomState(0x64192)

    # One channel per row (N_TG=1): 128-lane rows, N_TOK valid tokens each.
    x = np.zeros((N_CH, LANES), dtype=np.float32)
    x[:, :N_TOK] = rng.uniform(-4.0, 4.0, size=(N_CH, N_TOK))
    gamma = rng.uniform(0.5, 1.5, size=N_CH).astype(np.float32)
    beta = rng.uniform(-0.5, 0.5, size=N_CH).astype(np.float32)

    input_path = tmp_path / "input_x_fp32.bin"
    gamma_path = tmp_path / "gamma_fp32.bin"
    beta_path = tmp_path / "beta_fp32.bin"
    input_path.write_bytes(x.tobytes())
    gamma_path.write_bytes(gamma.tobytes())
    beta_path.write_bytes(beta.tobytes())
    output_path = tmp_path / "output.bin"

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    app = LayerNorm64x192App(
        inst_path=_INST_BIN,
        input_path=input_path,
        gamma_path=gamma_path,
        beta_path=beta_path,
        output_path=output_path,
    )
    state, cycles = app.run(max_cycles=5_000_000, state=state)
    assert cycles > 0

    expected = _reference_layernorm(x[:, :N_TOK], gamma, beta)

    raw = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    assert raw.size == N_CH * (ROW_BYTES // 4), (
        f"output has {raw.size} floats, expected {N_CH * (ROW_BYTES // 4)}"
    )
    # Crop each whole output row down to the valid token count.
    actual = raw.reshape(N_CH, ROW_BYTES // 4)[:, :N_TOK]

    np.testing.assert_allclose(
        actual, expected,
        rtol=1e-4, atol=1e-3,
        err_msg="LayerNorm 64x192 output does not match reference",
    )
