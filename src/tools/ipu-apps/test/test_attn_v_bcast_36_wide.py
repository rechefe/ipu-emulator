"""Wide-vector FP32 end-to-end test for attn_v_bcast_36.

Runs the REAL kernel binary against a numpy reference in wide-vector debug
mode. No checked-in golden: FP32 inputs are generated here and
O[i, t] = sum_s P[i, s] * V[s, t] is computed directly.

This is the key-major P variant of attn@V; `attn_v_256x36` is the query-major
+ AGG kernel and shares V's and O's layouts.

There is no AGG in this kernel: the contraction over all 256 keys is a single
continuous ACC.ADD (ACC.ADD.FIRST at s=0) per-lane float32 running sum,
rounded on every step -- not the AGG float64 left-fold `attn_v_256x36`'s
reference needs. The reference mirrors that per-step fold exactly
(`_acc_reference`), so it agrees with the kernel's output exactly, not merely
within a loose tolerance.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.attention.attn_v_bcast_36 import (
    AttnVBcast36App, N_TOK, D, N_HEAD, N_CHAN, LANES, ELEM_BYTES,
    PV_STRIDE_ROWS, P_HEAD_STRIDE_ROWS, ROW_BYTES,
)

ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/ipu_apps/attention/attn_v_bcast_36/attn_v_bcast_36.asm"
)


@pytest.fixture(scope="module")
def inst_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "attn_v_bcast_36.bin"
        assemble_to_bin_file(ASM_PATH.read_text(encoding="utf-8"), str(path))
        yield path


def _acc_reference(P: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Reference mirroring the emulator's single continuous ACC.ADD datapath.

    For each (head, channel t): a single s_loop over all 256 keys forms
    float32 lane products mult_res[i] = P[i,s] * V[s,t] (lane = query i, ACC.ADD.FIRST
    at s=0) and ACC.ADD writes the running sum back as float32 every step --
    one continuous float32 left-fold over all 256 keys, with no group split
    or reset (the 128/128 R0/R1 split is only how V's scalar source is
    staged, not a break in the accumulation).
    """
    out = np.zeros((N_HEAD, N_TOK, D), dtype=np.float32)
    for h in range(N_HEAD):
        for t in range(D):
            acc = np.zeros(N_TOK, dtype=np.float32)
            for s in range(N_TOK):
                prod = (P[h, :, s].astype(np.float32)
                        * np.float32(V[h, t, s])).astype(np.float32)
                acc = prod if s == 0 else (acc + prod).astype(np.float32)
            out[h, :, t] = acc
    return out


def test_attn_v_bcast_36_wide_fp32(inst_file: Path, tmp_path: Path) -> None:
    rng = np.random.RandomState(0xA17)

    # P[h, i, s] — attention probabilities; V[h, t, s] — values, channel-major.
    P = rng.uniform(-1.0, 1.0, size=(N_HEAD, N_TOK, N_TOK)).astype(np.float32)
    V = rng.uniform(-1.0, 1.0, size=(N_HEAD, D, N_TOK)).astype(np.float32)

    # P is staged KEY-major: P[i, s] at PBASE + h*P_HEAD_STRIDE + s*PV_STRIDE + i.
    p_buf = np.zeros((N_HEAD, N_TOK, PV_STRIDE_ROWS * LANES), dtype=np.float32)
    for h in range(N_HEAD):
        for s in range(N_TOK):
            p_buf[h, s, :N_TOK] = P[h, :, s]        # column s = all queries
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
    app = AttnVBcast36App(
        inst_path=inst_file,
        p_path=p_path,
        v_path=v_path,
        output_path=output_path,
    )
    state, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    # O[h, i, t] = sum_s P[h, i, s] * V[h, t, s], via the per-step ACC fold.
    expected = _acc_reference(P, V)                  # [N_HEAD, N_TOK, D]

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
                rtol=0, atol=0,
                err_msg=f"attn@V mismatch for head {h}, channel {t}",
            )
