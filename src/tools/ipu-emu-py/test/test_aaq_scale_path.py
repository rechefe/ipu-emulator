"""Probe the ACTIVATE.QUANTIZE data path.

Reconciled against ZDlinear+main (post AGG-mult_res fix, ACTIVATE+AAQ fusion):
1. ACTIVATE.QUANTIZE applies a FIXED per-lane direct INT8 clamp to [-128,127] —
   there is NO computed shared scale derived from a max/abs-max reduction over
   R_ACC. Each lane is clamped independently. (The earlier >>24 truncation was
   replaced upstream by this interim clamp.)
2. ACTIVATE.QUANTIZE is the only per-lane writer of POST_AAQ_REG, and it reads
   R_ACC. It replaced the old two-step ACTIVATE (wide staging) -> AAQ (quantize)
   pair: activation and the INT8 clamp now happen in one instruction, so the
   output bytes are INT8, not the old 32-bit staged lanes.
3. AGG.* reduces mult_res and writes the reduced scalar into an R_ACC lane
   (never POST_AAQ_REG).

The active lane count now comes from the dstructure CR named by the
instruction (cr_idx), replacing the removed full_xmem_row 0/1 flag.
"""

from __future__ import annotations

import struct

from ipu_common.activations import ACTIVATION_IDENTITY, ACTIVATION_RELU
from ipu_emu.ipu import Ipu
from ipu_emu.ipu_math import DType
from ipu_emu.ipu_state import IpuState


def _int8_state() -> IpuState:
    st = IpuState()  # default (non wide-vector) path
    st.dtype = DType.INT8
    return st


def _seed_acc(st: IpuState, values: list[int]) -> None:
    """Write ``values`` into the leading R_ACC lanes (int32 each)."""
    acc = bytearray(512)
    for i, v in enumerate(values):
        struct.pack_into("<i", acc, i * 4, v)
    st.regfile.set_r_acc_bytes(bytes(acc))


def test_quantize_is_fixed_clamp_not_a_shared_scale() -> None:
    """Two lanes with different magnitudes are each clamped independently.

    A real shared-scale quantizer would pick one scale from the vector max and
    apply it to all lanes. Here each lane is clamped to [-128,127] on its own,
    with no max/abs-max reduction feeding a shared scale: lane0=200 -> 127
    (clamped), lane1=5 -> 5 (in range). A shared scale keyed off lane0's
    magnitude would have pulled lane1 toward 0; it does not.

    Uses ``identity`` so the clamp is observed on its own.
    """
    st = _int8_state()
    ipu = Ipu(st)

    _seed_acc(st, [200, 5])  # 200 -> clamps to 127; 5 stays 5
    st.set_cr_dstructure(128)
    ipu.snapshot = st.regfile.snapshot()

    ipu.execute_activate_quantize(activation_fn=ACTIVATION_IDENTITY, cr_idx=15)

    out = st.regfile.raw("post_aaq_reg")
    assert struct.unpack_from("b", out, 0)[0] == 127
    assert struct.unpack_from("b", out, 1)[0] == 5


def test_quantize_extremes_map_to_int8_endpoints() -> None:
    """The largest/smallest int32 R_ACC lanes land on the INT8 endpoints, per lane.

    The point is that each endpoint is reached independently, not via any
    shared scale.
    """
    st = _int8_state()
    ipu = Ipu(st)

    _seed_acc(st, [0x7FFFFFFF, -0x80000000])  # +max -> 127, -max -> -128
    st.set_cr_dstructure(128)
    ipu.snapshot = st.regfile.snapshot()

    ipu.execute_activate_quantize(activation_fn=ACTIVATION_IDENTITY, cr_idx=15)

    out = st.regfile.raw("post_aaq_reg")
    assert struct.unpack_from("b", out, 0)[0] == 127
    assert struct.unpack_from("b", out, 1)[0] == -128


def test_activate_quantize_writes_post_aaq_per_lane_from_racc() -> None:
    """ACTIVATE.QUANTIZE is the only per-lane writer of POST_AAQ_REG, sourced from R_ACC.

    Post-fusion the written lanes are INT8 bytes (activation + clamp in one
    step), not the 32-bit staged lanes the old separate ACTIVATE produced.
    """
    st = _int8_state()
    ipu = Ipu(st)

    _seed_acc(st, [(i - 1) * 10 for i in range(4)])  # -10, 0, 10, 20
    st.set_cr_dstructure(128)
    ipu.snapshot = st.regfile.snapshot()

    ipu.execute_activate_quantize(activation_fn=ACTIVATION_RELU, cr_idx=15)

    post = st.regfile.raw("post_aaq_reg")
    vals = [struct.unpack_from("b", post, i)[0] for i in range(4)]
    assert vals == [0, 0, 10, 20]  # relu applied independently per lane


def test_agg_sum_writes_into_racc_not_post_aaq() -> None:
    """AGG.SUM.FIRST reduces mult_res; the scalar lands in R_ACC, POST_AAQ untouched."""
    st = _int8_state()
    ipu = Ipu(st)

    # Post-fix: AGG reduces mult_res. Seed mult_res lanes 0..3 = 1,2,3,4 (sum 10).
    mult = bytearray(512)
    for i in range(4):
        struct.pack_into("<i", mult, i * 4, i + 1)  # 1,2,3,4
    st.regfile.raw("mult_res")[:] = mult
    pre_post = bytes(st.regfile.raw("post_aaq_reg"))
    st.set_cr_dstructure(4)  # valid_elements = 4 -> sum mult_res lanes 0..3
    ipu.snapshot = st.regfile.snapshot()

    # dest_slot is the resolved destination value (LR lookup happens in dispatch).
    ipu.execute_agg_sum_first(dest_slot=7, cr_idx=15)

    # Reduced scalar (10) landed in R_ACC[7], and POST_AAQ_REG was untouched.
    acc_out = st.regfile.raw("r_acc")
    assert struct.unpack_from("<i", acc_out, 7 * 4)[0] == 10
    assert bytes(st.regfile.raw("post_aaq_reg")) == pre_post
