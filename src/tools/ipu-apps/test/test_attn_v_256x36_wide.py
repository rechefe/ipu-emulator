"""Wide-vector FP32 end-to-end test for attn_v_256x36.

Runs the REAL kernel binary against a numpy reference in wide-vector debug
mode. No checked-in golden: FP32 inputs are generated here and
O[i, t] = sum_s P[i, s] * V[s, t] is computed directly.

This is the query-major P + AGG variant of attn@V; `attn_v_bcast_36` is the
key-major broadcast kernel and shares V's and O's layouts.

L3 is the only shape with TWO key chunks (N_TOK=256 spans two 128-lane
groups), so it is the only place cross-chunk AGG accumulation is exercised.
The reference therefore mirrors AGG's actual datapath rather than calling
``np.einsum``: chunk 0 is a float64 left-fold over 128 lanes rounded to
float32 by ``AGG.SUM.FIRST``'s R_ACC write; chunk 1 is a second float64
left-fold added, in float32, to that already-rounded chunk-0 result by
``AGG.SUM`` (see ``execute_agg_sum``: it adds the new float64-folded partial
to the float32 snapshot of R_ACC, then rounds once more). A plain
``np.einsum`` accumulates in one pass with different rounding and cannot
discriminate a wrong cross-chunk carry from a right one.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.attention.attn_v_256x36 import (
    AttnV256x36App, N_TOK, D, N_HEAD, N_CHAN, LANES,
    PV_STRIDE_ROWS, P_HEAD_STRIDE_ROWS,
)

ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/ipu_apps/attention/attn_v_256x36/attn_v_256x36.asm"
)


@pytest.fixture(scope="module")
def inst_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "attn_v_256x36.bin"
        assemble_to_bin_file(ASM_PATH.read_text(encoding="utf-8"), str(path))
        yield path


def _agg_reference(P: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Reference mirroring the emulator's two-chunk AGG datapath.

    For each (head, channel t, query i): chunk 0 (keys 0..127) is a float64
    left-fold of the 128 float32 lane products, rounded to float32 once by
    AGG.SUM.FIRST's R_ACC write. Chunk 1 (keys 128..255) is a second float64
    left-fold of its own 128 lane products, added -- in float32, to the
    already-rounded chunk-0 value -- by AGG.SUM, then rounded once more.
    """
    out = np.zeros((N_HEAD, N_TOK, D), dtype=np.float32)
    for h in range(N_HEAD):
        for t in range(D):
            v = V[h, t].astype(np.float32)              # [N_TOK] keys
            for i in range(N_TOK):
                lanes = (P[h, i].astype(np.float32) * v).astype(np.float32)

                total0 = 0.0
                for x in lanes[:128]:
                    total0 += float(x)
                chunk0 = np.float32(total0)

                total1 = 0.0
                for x in lanes[128:]:
                    total1 += float(x)
                chunk1 = np.float32(total1)

                out[h, i, t] = np.float32(float(chunk1) + float(chunk0))
    return out


def test_attn_v_256x36_wide_fp32(inst_file: Path, tmp_path: Path) -> None:
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
        inst_path=inst_file,
        p_path=p_path,
        v_path=v_path,
        output_path=output_path,
    )
    state, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    # O[h, i, t] = sum_s P[h, i, s] * V[h, t, s], via the two-chunk AGG fold.
    expected = _agg_reference(P, V)                  # [N_HEAD, N_TOK, D]

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
                rtol=1e-6, atol=1e-6,
                err_msg=f"attn@V mismatch for head {h}, channel {t}",
            )
