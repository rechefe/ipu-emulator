"""Wide-vector FP32 end-to-end test for layernorm_16x240 (Layer 5).

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

from ipu_apps.layernorm_16x240 import (
    LayerNorm16x240App, N_CH, N_TOK, LANES,
)

_INST_BIN = Path(os.environ["LAYERNORM_16X240_INST_BIN"])


def _reference_layernorm(
    x: np.ndarray, gamma: np.ndarray, beta: np.ndarray
) -> np.ndarray:
    """output[ch, i] = gamma[ch] * (x[ch,i] - mu[i]) / sigma[i] + beta[ch].

    mu/sigma reduce over the CHANNEL axis, independently per token -- the
    normalizer is 1/N_CH = 1/240, NOT 1/N_TOK. No epsilon: this matches the
    rsqrt activation, which handles stability internally.

    x: [N_CH, N_TOK], gamma/beta: [N_CH]  ->  [N_CH, N_TOK]
    """
    mean = x.mean(axis=0)                                   # [N_TOK]
    centered = x - mean                                     # [N_CH, N_TOK]
    var = (centered ** 2).mean(axis=0)                      # [N_TOK]
    inv_std = np.where(var > 0.0, 1.0 / np.sqrt(var), 0.0)
    normalized = centered * inv_std
    return (gamma[:, None] * normalized + beta[:, None]).astype(np.float32)


def test_layernorm_16x240_wide_fp32(tmp_path: Path) -> None:
    rng = np.random.RandomState(0x16240)

    # One channel per row (N_TG=1): 128-lane rows, N_TOK=16 valid tokens each.
    # The channel owns the WHOLE row; lanes 16..127 are zero padding.
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
    app = LayerNorm16x240App(
        inst_path=_INST_BIN,
        input_path=input_path,
        gamma_path=gamma_path,
        beta_path=beta_path,
        output_path=output_path,
    )
    state, cycles = app.run(max_cycles=5_000_000, state=state)
    assert cycles > 0

    expected = _reference_layernorm(x[:, :N_TOK], gamma, beta)

    # teardown() already cropped each row to its N_TOK valid tokens.
    actual = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    assert actual.size == N_CH * N_TOK, (
        f"output has {actual.size} floats, expected {N_CH * N_TOK}"
    )
    actual = actual.reshape(N_CH, N_TOK)

    np.testing.assert_allclose(
        actual, expected,
        rtol=1e-4, atol=1e-3,
        err_msg="LayerNorm 16x240 output does not match reference",
    )

    # ---- Guards against a degenerate pass -----------------------------------
    # A kernel that emitted all zeros, or that passed its input straight
    # through, could not satisfy these.

    # 1. The output actually varies, and varies the way the reference does.
    assert actual.std() > 0.1, "output is (near) constant -- degenerate result"
    np.testing.assert_allclose(actual.std(), expected.std(), rtol=1e-3)

    # 2. Real LayerNorm centering: undo gamma/beta and the per-channel scale,
    #    then the mean over CHANNELS must be ~0 for every token, and the
    #    standard deviation over channels must be ~1.
    de_scaled = (actual - beta[:, None]) / gamma[:, None]
    np.testing.assert_allclose(
        de_scaled.mean(axis=0), np.zeros(N_TOK), atol=1e-4,
        err_msg="de-scaled output is not centred over the channel axis",
    )
    np.testing.assert_allclose(
        de_scaled.std(axis=0), np.ones(N_TOK), rtol=1e-3,
        err_msg="de-scaled output does not have unit variance over channels",
    )

    # 3. The output is genuinely different from the input, so a pass-through
    #    of x could not have produced it.
    assert not np.allclose(actual, x[:, :N_TOK], rtol=1e-2, atol=1e-2), (
        "output equals the input -- the kernel is a pass-through"
    )
