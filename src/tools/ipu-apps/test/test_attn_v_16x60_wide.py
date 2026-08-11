"""Wide-vector FP32 end-to-end test for attn_v_16x60 (Layer 5).

Runs the REAL kernel binary against a numpy reference in wide-vector debug
mode. No checked-in golden: FP32 inputs are generated here and
O[i, t] = sum_s P[i, s] * V[s, t] is computed directly.

This is the query-major P + AGG variant of attn@V, the second half of the
QUERY-MAJOR chain (``qk_scores_16x60`` -> ``attn_v_16x60``).  ``attn_v_bcast_60``
is the key-major broadcast kernel; it shares V's and O's layouts but is a
DIFFERENT mapping with its own golden -- the two chains are bit-different by
design and must never share expectations.

The reference mirrors AGG's datapath exactly (see ``_agg_reference``): a plain
float32 dot product is NOT the same computation.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.attn_v_16x60 import (
    AttnV16x60App, N_TOK, D, N_HEAD, N_CHAN, LANES,
    PV_STRIDE_ROWS, P_HEAD_STRIDE_ROWS,
)

_INST_BIN = Path(os.environ["ATTN_V_16X60_INST_BIN"])


def _agg_reference(P: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Reference for O[h,i,t] that mirrors the emulator's AGG datapath.

    ``AGG.SUM.FIRST`` (ipu.py ``_agg_sum_lanes``) left-folds the active
    MULT_RES lanes starting from a Python float (float64) and then rounds the
    total ONCE to float32 on the R_ACC write.  The per-lane products themselves
    are float32 (MULT writes ``<f`` lanes).  At L5 all 16 keys are one chunk, so
    there is exactly one fold and one rounding -- no cross-chunk partial.

    A plain ``np.einsum`` in float32 would accumulate in a different order and
    round every step, so it is not the same computation.
    """
    out = np.zeros((N_HEAD, N_TOK, D), dtype=np.float32)
    for h in range(N_HEAD):
        for t in range(D):
            for i in range(N_TOK):
                # float32 element-wise products (MULT_RES lanes)...
                lanes = (P[h, i, :].astype(np.float32) * V[h, t, :].astype(np.float32))
                # ...left-folded in float64, rounded once on the R_ACC write.
                total = 0.0
                for s in range(N_TOK):
                    total += float(lanes[s])
                out[h, i, t] = np.float32(total)
    return out


def test_attn_v_16x60_wide_fp32(tmp_path: Path) -> None:
    rng = np.random.RandomState(0xA60)

    # P[h, i, s] — attention probabilities; V[h, t, s] — values, channel-major.
    P = rng.uniform(-1.0, 1.0, size=(N_HEAD, N_TOK, N_TOK)).astype(np.float32)
    V = rng.uniform(-1.0, 1.0, size=(N_HEAD, D, N_TOK)).astype(np.float32)

    # P is staged QUERY-major, one WHOLE row per query (16 live lanes of 128).
    p_buf = np.zeros((N_HEAD, N_TOK, PV_STRIDE_ROWS * LANES), dtype=np.float32)
    for h in range(N_HEAD):
        p_buf[h, :, :N_TOK] = P[h]                  # row i = all keys for query i
    assert p_buf[0].size == P_HEAD_STRIDE_ROWS * LANES

    # V is channel-major, one WHOLE row per value channel.
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
    app = AttnV16x60App(
        inst_path=_INST_BIN,
        p_path=p_path,
        v_path=v_path,
        output_path=output_path,
    )
    state, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    expected = _agg_reference(P, V)                 # [N_HEAD, N_TOK, D]

    # Output: channel (h*60 + t) occupies ONE whole row; query i is lane i.
    raw = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    assert raw.size == N_CHAN * LANES, (
        f"output has {raw.size} floats, expected {N_CHAN * LANES}"
    )
    got = raw.reshape(N_CHAN, LANES)                # [chan, query lane]

    for h in range(N_HEAD):
        for t in range(D):
            np.testing.assert_allclose(
                got[h * D + t, :N_TOK], expected[h, :, t],
                rtol=1e-4, atol=1e-3,
                err_msg=f"attn@V mismatch for head {h}, channel {t}",
            )


def test_attn_v_16x60_wide_fp32_padding_is_inert(tmp_path: Path) -> None:
    """AGG.SUM.FIRST reduces across lanes, so the trailing 112 padding lanes
    of every P and V row must be excluded by valid_elements, not by relying on
    the harness's zero-fill. Refill them with garbage (a chained producer
    would leave another stream's data there) and assert the valid-lane output
    is bit-identical to the all-zero-padding run.
    """
    rng = np.random.RandomState(0xA60)

    P = rng.uniform(-1.0, 1.0, size=(N_HEAD, N_TOK, N_TOK)).astype(np.float32)
    V = rng.uniform(-1.0, 1.0, size=(N_HEAD, D, N_TOK)).astype(np.float32)

    def run(pad_value: float) -> np.ndarray:
        p_buf = np.full((N_HEAD, N_TOK, PV_STRIDE_ROWS * LANES), pad_value, dtype=np.float32)
        for h in range(N_HEAD):
            p_buf[h, :, :N_TOK] = P[h]

        v_buf = np.full((N_CHAN, PV_STRIDE_ROWS * LANES), pad_value, dtype=np.float32)
        for h in range(N_HEAD):
            for t in range(D):
                v_buf[h * D + t, :N_TOK] = V[h, t, :]

        p_path = tmp_path / f"p_fp32_{pad_value}.bin"
        v_path = tmp_path / f"v_fp32_{pad_value}.bin"
        p_path.write_bytes(p_buf.tobytes())
        v_path.write_bytes(v_buf.tobytes())
        output_path = tmp_path / f"output_{pad_value}.bin"

        state = IpuState(
            wide_vector_debug=True,
            wide_vector_arithmetic=WideVectorArithmetic.FP32,
        )
        app = AttnV16x60App(
            inst_path=_INST_BIN,
            p_path=p_path,
            v_path=v_path,
            output_path=output_path,
        )
        state, cycles = app.run(max_cycles=20_000_000, state=state)
        assert cycles > 0

        raw = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
        return raw.reshape(N_CHAN, LANES)[:, :N_TOK]

    zero_padded = run(0.0)
    garbage_padded = run(1e3)

    np.testing.assert_array_equal(
        garbage_padded, zero_padded,
        err_msg=(
            "attn_v_16x60 output changed when padding lanes were filled with "
            "garbage -- AGG reduction is not structurally isolated from "
            "unused lanes"
        ),
    )
