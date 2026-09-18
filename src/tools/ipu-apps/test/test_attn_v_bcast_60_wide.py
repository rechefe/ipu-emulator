"""Wide-vector FP32 end-to-end test for attn_v_bcast_60 (Layer 5).

Runs the REAL kernel binary against a numpy reference in wide-vector debug
mode. No checked-in golden: FP32 inputs are generated here and
O[i, t] = sum_s P[i, s] * V[s, t] is computed directly.

This is the key-major P + BROADCAST variant of attn@V, the second half of the
KEY-MAJOR chain (``attn_scores_km_16x60`` -> ``attn_v_bcast_60``).  The
query-major sibling ``attn_v_16x60`` computes the same mathematical product
with a DIFFERENT datapath (MULT.RC.VV + AGG.SUM); the two chains are
bit-different BY DESIGN and must never share expectations, so this reference is
computed for the BROADCAST mapping here.

There is no AGG in this kernel: the contraction is ACC.ADD over 16 keys, which
rounds to float32 on every step (``execute_acc_add`` packs ``<f`` each cycle).
``_acc_reference`` mirrors that exactly -- the AGG float64 left-fold used by
``attn_v_16x60``'s reference would be the WRONG datapath here.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.attention.attn_v_bcast_60 import (
    AttnVBcast60App, N_TOK, D, N_HEAD, N_CHAN, LANES,
    PV_STRIDE_ROWS, P_HEAD_STRIDE_ROWS,
)

ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/ipu_apps/attention/attn_v_bcast_60/attn_v_bcast_60.asm"
)


@pytest.fixture(scope="module")
def inst_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "attn_v_bcast_60.bin"
        assemble_to_bin_file(ASM_PATH.read_text(encoding="utf-8"), str(path))
        yield path


def _acc_reference(P: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Reference for O[h,i,t] mirroring the emulator's MULT + ACC.ADD datapath.

    Each cycle produces float32 products ``mult_res[i] = P[i,s] * V[s,t]``
    (MULT writes ``<f`` lanes) and ``ACC.ADD`` writes the running sum back as
    float32.  So the contraction is a float32 left-fold over keys, rounded at
    every step -- not a float64 accumulation and not AGG's single-rounding fold.
    """
    out = np.zeros((N_HEAD, N_TOK, D), dtype=np.float32)
    for h in range(N_HEAD):
        for t in range(D):
            acc = np.zeros(N_TOK, dtype=np.float32)
            for s in range(N_TOK):
                prod = (P[h, :, s].astype(np.float32)
                        * np.float32(V[h, t, s]))       # mult_res lanes
                acc = (acc + prod).astype(np.float32) if s else prod.astype(np.float32)
            out[h, :, t] = acc
    return out


def test_attn_v_bcast_60_wide_fp32(inst_file: Path, tmp_path: Path) -> None:
    rng = np.random.RandomState(0xB60)

    # P[h, i, s] — attention probabilities; V[h, t, s] — values, channel-major.
    P = rng.uniform(-1.0, 1.0, size=(N_HEAD, N_TOK, N_TOK)).astype(np.float32)
    V = rng.uniform(-1.0, 1.0, size=(N_HEAD, D, N_TOK)).astype(np.float32)

    # P is staged KEY-major: one WHOLE row per key, query i in lane i.
    p_buf = np.zeros((N_HEAD, N_TOK, PV_STRIDE_ROWS * LANES), dtype=np.float32)
    for h in range(N_HEAD):
        p_buf[h, :, :N_TOK] = P[h].T                # row s = all queries for key s
    assert p_buf[0].size == P_HEAD_STRIDE_ROWS * LANES

    # V is channel-major, one WHOLE row per value channel (same as attn_v_16x60).
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
    app = AttnVBcast60App(
        inst_path=inst_file,
        p_path=p_path,
        v_path=v_path,
        output_path=output_path,
    )
    state, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    expected = _acc_reference(P, V)                 # [N_HEAD, N_TOK, D]

    # Output: channel (h*60 + t) occupies ONE whole row; query i is lane i --
    # the SAME O layout as attn_v_16x60.
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
                err_msg=f"attn@V (broadcast) mismatch for head {h}, channel {t}",
            )


def test_attn_v_bcast_60_wide_fp32_padding_is_inert(inst_file: Path, tmp_path: Path) -> None:
    """No AGG: ACC.ADD accumulates each of the 128 lanes (queries)
    independently, so the 112 padding lanes of P and V can only ever waste
    lanes, never contaminate the 16 valid ones. Prove it: refill the padding
    with garbage and assert the valid-lane output is bit-identical.
    """
    rng = np.random.RandomState(0xB60)

    P = rng.uniform(-1.0, 1.0, size=(N_HEAD, N_TOK, N_TOK)).astype(np.float32)
    V = rng.uniform(-1.0, 1.0, size=(N_HEAD, D, N_TOK)).astype(np.float32)

    def run(pad_value: float) -> np.ndarray:
        p_buf = np.full((N_HEAD, N_TOK, PV_STRIDE_ROWS * LANES), pad_value, dtype=np.float32)
        for h in range(N_HEAD):
            p_buf[h, :, :N_TOK] = P[h].T

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
        app = AttnVBcast60App(
            inst_path=inst_file,
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
            "attn_v_bcast_60 output changed when padding lanes were filled "
            "with garbage -- per-lane ACC is not isolated from unused lanes"
        ),
    )
