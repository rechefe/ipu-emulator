"""Wide-vector FP32 end-to-end test for attn_v_256x36.

Runs the REAL kernel binary against a numpy reference in wide-vector debug
mode. No checked-in golden: FP32 inputs are generated here and
O[i, t] = sum_s P[i, s] * V[s, t] is computed directly.

This is the query-major P + AGG variant of attn@V; `attn_v_bcast_36` is the
key-major broadcast kernel and shares V's and O's layouts.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.attn_v_256x36 import (
    AttnV256x36App, N_TOK, D, N_HEAD, N_CHAN, LANES,
    PV_STRIDE_ROWS, P_HEAD_STRIDE_ROWS,
)

_INST_BIN = Path(os.environ["ATTN_V_256X36_INST_BIN"])


def test_attn_v_256x36_wide_fp32(tmp_path: Path) -> None:
    rng = np.random.RandomState(0xA18)

    # P[h, i, s] — attention probabilities; V[h, t, s] — values, channel-major.
    P = rng.uniform(-1.0, 1.0, size=(N_HEAD, N_TOK, N_TOK)).astype(np.float32)
    V = rng.uniform(-1.0, 1.0, size=(N_HEAD, D, N_TOK)).astype(np.float32)

    # P is staged QUERY-major: P[i, s] at PBASE + h*P_HEAD_STRIDE + i*PV_STRIDE + s.
    p_buf = np.zeros((N_HEAD, N_TOK, PV_STRIDE_ROWS * LANES), dtype=np.float32)
    for h in range(N_HEAD):
        p_buf[h, :, :N_TOK] = P[h]                  # row i = all keys for query i
    assert p_buf[0].size == P_HEAD_STRIDE_ROWS * LANES

    # V is channel-major: V[s, chan] at VBASE + chan*PV_STRIDE + s.
    v_buf = np.zeros((N_CHAN, PV_STRIDE_ROWS * LANES), dtype=np.float32)
    for h in range(N_HEAD):
        for t in range(D):
            v_buf[h * D + t, :N_TOK] = V[h, t, :]

    p_path = tmp_path / "p_fp32.bin"
    v_path = tmp_path / "v_fp32.bin"
    p_path.write_bytes(p_buf.tobytes())
    v_path.write_bytes(v_buf.tobytes())
    output_path = tmp_path / "output.bin"

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    app = AttnV256x36App(
        inst_path=_INST_BIN,
        p_path=p_path,
        v_path=v_path,
        output_path=output_path,
    )
    state, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    # O[h, i, t] = sum_s P[h, i, s] * V[h, t, s]
    expected = np.einsum("his,hts->hit", P, V)      # [N_HEAD, N_TOK, D]

    # Output: channel (h*36 + t) occupies 2 group rows of LANES FP32 lanes;
    # query i = g*LANES + local.
    raw = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    assert raw.size == N_CHAN * 2 * LANES, (
        f"output has {raw.size} floats, expected {N_CHAN * 2 * LANES}"
    )
    got = raw.reshape(N_CHAN, 2 * LANES)            # [chan, query]

    for h in range(N_HEAD):
        for t in range(D):
            np.testing.assert_allclose(
                got[h * D + t, :N_TOK], expected[h, :, t],
                rtol=1e-4, atol=1e-3,
                err_msg=f"attn@V mismatch for head {h}, channel {t}",
            )
