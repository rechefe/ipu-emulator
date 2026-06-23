"""Probe the AAQ/ACTIVATE quantization data path.

Reconciled against ZDlinear+main (post AGG-mult_res fix, AAQ clamp):
1. AAQ applies a FIXED per-lane direct INT8 clamp to [-128,127] (ipu.py:1010) —
   there is NO computed shared scale derived from a max/abs-max reduction over
   R_ACC. Each lane is clamped independently. (The earlier >>24 truncation was
   replaced upstream by this interim clamp.)
2. ACTIVATE writes R_ACC -> POST_AAQ_REG lane-for-lane (per-lane path exists
   for writing POST_AAQ_REG, sourced only from R_ACC).
3. AGG.* reduces mult_res and writes the reduced scalar into an R_ACC lane
   (never POST_AAQ_REG).
"""

from __future__ import annotations

import struct

from ipu_emu.ipu import Ipu
from ipu_emu.ipu_math import DType
from ipu_emu.ipu_state import IpuState


def _int8_state() -> IpuState:
    st = IpuState()  # default (non wide-vector) path
    st.dtype = DType.INT8
    return st


def test_aaq_is_fixed_clamp_not_a_shared_scale() -> None:
    """Two lanes with different magnitudes are each clamped independently.

    A real shared-scale quantizer would pick one scale from the vector max and
    apply it to all lanes. Here each lane is clamped to [-128,127] on its own,
    with no max/abs-max reduction feeding a shared scale: lane0=200 -> 127
    (clamped), lane1=5 -> 5 (in range). A shared scale keyed off lane0's
    magnitude would have pulled lane1 toward 0; it does not.
    """
    st = _int8_state()
    ipu = Ipu(st)

    post = bytearray(512)
    struct.pack_into("<i", post, 0, 200)  # out of range -> clamps to 127
    struct.pack_into("<i", post, 4, 5)    # in range -> stays 5
    st.regfile.set_post_aaq_reg(bytes(post))
    ipu.snapshot = st.regfile.snapshot()

    ipu.execute_aaq(full_xmem_row=1)

    out = st.regfile.raw("post_aaq_reg")
    assert struct.unpack_from("b", out, 0)[0] == 127
    assert struct.unpack_from("b", out, 1)[0] == 5


def test_aaq_extremes_map_to_int8_endpoints() -> None:
    """Largest/smallest int32 lanes >>24 land at the INT8 endpoints, per lane.

    With a single fixed 24-bit shift the truncated value is already inside
    [-128, 127] for any int32, so the clamp is defensive — the point here is
    that the endpoints are reached independently, not via any shared scale.
    """
    st = _int8_state()
    ipu = Ipu(st)
    post = bytearray(512)
    struct.pack_into("<i", post, 0, 0x7FFFFFFF)            # +max -> 127
    struct.pack_into("<i", post, 4, -0x80000000)           # -max -> -128
    st.regfile.set_post_aaq_reg(bytes(post))
    ipu.snapshot = st.regfile.snapshot()

    ipu.execute_aaq(full_xmem_row=1)

    out = st.regfile.raw("post_aaq_reg")
    assert struct.unpack_from("b", out, 0)[0] == 127
    assert struct.unpack_from("b", out, 1)[0] == -128


def test_activate_writes_post_aaq_per_lane_from_racc() -> None:
    """ACTIVATE is the only per-lane writer of POST_AAQ_REG, sourced from R_ACC."""
    st = _int8_state()
    ipu = Ipu(st)

    acc = bytearray(512)
    for i in range(4):
        struct.pack_into("<i", acc, i * 4, (i - 1) * 10)  # -10, 0, 10, 20
    st.regfile.set_r_acc_bytes(bytes(acc))
    ipu.snapshot = st.regfile.snapshot()

    # activation_fn=1 == relu in the enum
    ipu.execute_activate(activation_fn=1, full_xmem_row=1)

    post = st.regfile.raw("post_aaq_reg")
    vals = [struct.unpack_from("<i", post, i * 4)[0] for i in range(4)]
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
    ipu.execute_agg_sum_first(dest_slot=7, full_xmem_row=0)

    # Reduced scalar (10) landed in R_ACC[7], and POST_AAQ_REG was untouched.
    acc_out = st.regfile.raw("r_acc")
    assert struct.unpack_from("<i", acc_out, 7 * 4)[0] == 10
    assert bytes(st.regfile.raw("post_aaq_reg")) == pre_post
