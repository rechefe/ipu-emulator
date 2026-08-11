"""Wide-vector FP32 end-to-end test for attn_v_64x48.

Runs the REAL kernel binary against a numpy reference in wide-vector debug
mode. No checked-in golden: FP32 inputs are generated here and
O[b, i, t] = sum_s P[b, i, s] * V[b, s, t] is computed directly.

This is the QUERY-major P + AGG variant of attn@V; it pairs with
``qk_scores_64x48``. ``attn_v_bcast_48`` is the key-major broadcast kernel and
shares V's and O's layouts, but produces bit-different results by design and
carries its own golden -- the two must never share one.

The golden mirrors AGG's datapath rather than calling einsum: the emulator's
``_agg_sum_lanes`` accumulates the 128 MULT_RES lanes as a Python float
(float64) left-fold and ``struct.pack_into("<f", ...)`` rounds that sum to
float32 exactly once, on the R_ACC write. A plain per-key float32 fold would
have a different rounding profile.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.attn_v_64x48 import (
    AttnV64x48App, N_TOK, D, N_BLOCK, N_CHAN, LANES,
)

_INST_BIN = Path(os.environ["ATTN_V_64X48_INST_BIN"])


def _agg_reference(P: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Reference mirroring the emulator's AGG datapath.

    For each (block, channel t, query i): MULT.RC.VV forms 128 float32 lane
    products (the trailing 64 lanes are zero padding), AGG.SUM.FIRST reduces
    them with a float64 left-fold, and the single R_ACC write rounds the total
    to float32. There is exactly ONE key chunk at N_TOK = 64, so no cross-chunk
    partial is carried.
    """
    out = np.zeros((N_BLOCK, D, N_TOK), dtype=np.float32)
    for b in range(N_BLOCK):
        for t in range(D):
            v = V[b, t].astype(np.float32)                  # [N_TOK] keys
            for i in range(N_TOK):
                # Lane products are computed in float32 (the emulator packs
                # each MULT_RES lane as "<f").
                lanes = (P[b, i].astype(np.float32) * v).astype(np.float32)
                # float64 left-fold over the live lanes, then one float32 round.
                total = 0.0
                for x in lanes:
                    total += float(x)
                out[b, t, i] = np.float32(total)
    return out


def test_attn_v_64x48_wide_fp32(tmp_path: Path) -> None:
    rng = np.random.RandomState(0xA48)

    # P[b, i, s] — attention probabilities (query-major); V[b, t, s] — values.
    P = rng.uniform(-1.0, 1.0, size=(N_BLOCK, N_TOK, N_TOK)).astype(np.float32)
    V = rng.uniform(-1.0, 1.0, size=(N_BLOCK, D, N_TOK)).astype(np.float32)

    # P is staged QUERY-major: one whole row per query, keys in leading lanes.
    p_buf = np.zeros((N_BLOCK * N_TOK, LANES), dtype=np.float32)
    for b in range(N_BLOCK):
        p_buf[b * N_TOK:(b + 1) * N_TOK, :N_TOK] = P[b]

    # V is channel-major: one whole row per value channel, keys leading.
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
    app = AttnV64x48App(
        inst_path=_INST_BIN,
        p_path=p_path,
        v_path=v_path,
        output_path=output_path,
    )
    state, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    expected = _agg_reference(P, V)                  # [N_BLOCK, D, N_TOK]

    # Output: channel (b*D + t) is one cropped row of N_TOK FP32 queries.
    raw = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    assert raw.size == N_CHAN * N_TOK, (
        f"output has {raw.size} floats, expected {N_CHAN * N_TOK}"
    )
    got = raw.reshape(N_BLOCK, D, N_TOK)             # [block, channel, query]

    max_err = float(np.max(np.abs(got - expected)))
    print(f"attn_v_64x48 max abs error vs AGG-datapath golden = {max_err:.3e}")

    np.testing.assert_allclose(
        got, expected, rtol=1e-4, atol=1e-3,
        err_msg="attn@V (query-major + AGG) mismatch",
    )


def test_attn_v_64x48_wide_fp32_padding_is_inert(tmp_path: Path) -> None:
    """AGG.SUM.FIRST reduces across lanes, so padding must be excluded by
    valid_elements rather than by the harness happening to zero-fill it.

    Refill the trailing 64 (unused) lanes of every P and V row with garbage
    (a real producer in a chained pipeline would leave another stream's data
    there, not zeros) and assert the valid-lane output is bit-identical to
    the all-zero-padding run. This proves the result does not depend on what
    is in the padding lanes.
    """
    rng = np.random.RandomState(0xA48)

    P = rng.uniform(-1.0, 1.0, size=(N_BLOCK, N_TOK, N_TOK)).astype(np.float32)
    V = rng.uniform(-1.0, 1.0, size=(N_BLOCK, D, N_TOK)).astype(np.float32)

    def run(pad_value: float) -> np.ndarray:
        p_buf = np.full((N_BLOCK * N_TOK, LANES), pad_value, dtype=np.float32)
        for b in range(N_BLOCK):
            p_buf[b * N_TOK:(b + 1) * N_TOK, :N_TOK] = P[b]

        v_buf = np.full((N_CHAN, LANES), pad_value, dtype=np.float32)
        for b in range(N_BLOCK):
            v_buf[b * D:(b + 1) * D, :N_TOK] = V[b]

        p_path = tmp_path / f"p_fp32_{pad_value}.bin"
        v_path = tmp_path / f"v_fp32_{pad_value}.bin"
        p_path.write_bytes(p_buf.tobytes())
        v_path.write_bytes(v_buf.tobytes())
        output_path = tmp_path / f"output_{pad_value}.bin"

        state = IpuState(
            wide_vector_debug=True,
            wide_vector_arithmetic=WideVectorArithmetic.FP32,
        )
        app = AttnV64x48App(
            inst_path=_INST_BIN,
            p_path=p_path,
            v_path=v_path,
            output_path=output_path,
        )
        state, cycles = app.run(max_cycles=20_000_000, state=state)
        assert cycles > 0

        raw = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
        return raw.reshape(N_BLOCK, D, N_TOK)

    zero_padded = run(0.0)
    garbage_padded = run(1e3)

    np.testing.assert_array_equal(
        garbage_padded, zero_padded,
        err_msg=(
            "attn_v_64x48 output changed when padding lanes were filled with "
            "garbage -- AGG reduction is not structurally isolated from "
            "unused lanes"
        ),
    )
