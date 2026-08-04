"""Wide-vector FP32 end-to-end test for attn_v_bcast_48.

Runs the REAL kernel binary against an inline numpy reference in wide-vector
debug mode. No checked-in golden: FP32 inputs are generated here and
O[b, i, t] = sum_s P[b, i, s] * V[b, s, t] is computed directly.

This is the KEY-major P + broadcast variant of attn@V; it pairs with
``attn_scores_km_64x48``. ``attn_v_64x48`` is the query-major + AGG kernel and
shares V's and O's layouts, but produces bit-different results by design and
carries its own golden -- the two must never share one.

This kernel uses **no AGG**: ``MULT.RC.VE`` broadcasts the scalar V[s,t] over
the key-major score row and ``ACC.ADD[.FIRST]`` runs a per-lane float32
accumulation over the 64 keys (see ``execute_acc_add``: one float32 add per lane
per key). The golden therefore mirrors a plain float32 running sum -- NOT the
float64 left-fold that the AGG sibling's golden needs.

Results are read back through the kernel's own store path
(ACTIVATE.QUANTIZE + STR_POST_AAQ_REG -> XMEM -> teardown crop), never R_ACC.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.attn_v_bcast_48 import (
    AttnVBcast48App, N_TOK, D, N_BLOCK, N_CHAN, LANES,
)

_INST_BIN = Path(os.environ["ATTN_V_BCAST_48_INST_BIN"])


def _acc_reference(P: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Reference mirroring the emulator's broadcast + ACC datapath.

    For each (block, channel t) the kernel walks keys s = 0..63, forming the
    float32 lane products P[i, s] * V[s, t] and adding them into R_ACC lane i in
    float32 (ACC.ADD.FIRST seeds, ACC.ADD accumulates). That is a plain float32
    running sum over keys -- reproduced here in the same order.
    """
    out = np.zeros((N_BLOCK, D, N_TOK), dtype=np.float32)
    for b in range(N_BLOCK):
        for t in range(D):
            acc = np.zeros(N_TOK, dtype=np.float32)
            for s in range(N_TOK):
                scalar = np.float32(V[b, t, s])
                prod = (P[b, :, s].astype(np.float32) * scalar).astype(np.float32)
                acc = (acc + prod).astype(np.float32) if s else prod
            out[b, t] = acc
    return out


def test_attn_v_bcast_48_wide_fp32(tmp_path: Path) -> None:
    rng = np.random.RandomState(0xB48)

    # P[b, i, s] — attention probabilities; V[b, t, s] — values (channel-major).
    P = rng.uniform(-1.0, 1.0, size=(N_BLOCK, N_TOK, N_TOK)).astype(np.float32)
    V = rng.uniform(-1.0, 1.0, size=(N_BLOCK, D, N_TOK)).astype(np.float32)

    # P is staged KEY-major: one whole row per key, holding that key's 64 query
    # scores in the leading lanes -- exactly attn_scores_km_64x48's output.
    p_buf = np.zeros((N_BLOCK * N_TOK, LANES), dtype=np.float32)
    for b in range(N_BLOCK):
        p_buf[b * N_TOK:(b + 1) * N_TOK, :N_TOK] = P[b].T      # [key, query]

    # V is channel-major: one whole row per value channel, keys leading.
    # Identical to attn_v_64x48's V layout.
    v_buf = np.zeros((N_CHAN, LANES), dtype=np.float32)
    for b in range(N_BLOCK):
        v_buf[b * D:(b + 1) * D, :N_TOK] = V[b]

    p_path = tmp_path / "p_fp32.bin"
    v_path = tmp_path / "v_fp32.bin"
    p_path.write_bytes(p_buf.tobytes())
    v_path.write_bytes(v_buf.tobytes())
    output_path = tmp_path / "output.bin"

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    app = AttnVBcast48App(
        inst_path=_INST_BIN,
        p_path=p_path,
        v_path=v_path,
        output_path=output_path,
    )
    state, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0
    print(f"attn_v_bcast_48: {cycles} cycles")
    print(state.stats.format_summary())

    expected = _acc_reference(P, V)                  # [N_BLOCK, D, N_TOK]

    # Output: channel (b*D + t) is one cropped row of N_TOK FP32 queries --
    # the same shape attn_v_64x48 emits.
    raw = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    assert raw.size == N_CHAN * N_TOK, (
        f"output has {raw.size} floats, expected {N_CHAN * N_TOK}"
    )
    got = raw.reshape(N_BLOCK, D, N_TOK)             # [block, channel, query]

    max_err = float(np.max(np.abs(got - expected)))
    print(f"attn_v_bcast_48 max abs error vs ACC-datapath golden = {max_err:.3e}")

    np.testing.assert_allclose(
        got, expected, rtol=1e-4, atol=1e-3,
        err_msg="attn@V (key-major + broadcast) mismatch",
    )
