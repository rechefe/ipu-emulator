"""Verification of R_ACC lane-level behavior for per-query scratch re-init.

Wide-vector FP32 mode is used for clarity (one float per lane, no quantization).
Goal: confirm which ACC-slot ops touch which R_ACC lanes, so a partial scratch
re-init does not clobber stashed outputs in the upper lanes.

NOTE on masking: ``_mult_mask_and_shift`` is a no-op in wide-vector mode
(ipu.py:378-379). The R_MASK machinery does not gate lanes in wide debug mode.
To get "MULT_RES[64:127] == 0 entering ACC" here we drive the multiply operand
lanes 64:127 to 0.0 (multiply-by-zero), which is the wide-mode equivalent and
exercises the exact same downstream ACC code paths.
"""

from __future__ import annotations

import struct

from ipu_emu.emulator import load_program, run_until_complete
from ipu_emu.execute import decode_instruction_word
from ipu_emu.ipu_math import DType
from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_as.lark_tree import assemble


SENTINEL = 7.0  # distinctive prior value parked in upper R_ACC lanes


def _new_state() -> IpuState:
    st = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    st.dtype = DType.INT8
    return st


def _load_run(st: IpuState, asm: str) -> None:
    encoded = assemble(asm)
    load_program(st, [decode_instruction_word(w) for w in encoded])
    run_until_complete(st)


def _acc_lanes(st: IpuState) -> list[float]:
    raw = st.regfile.raw("r_acc")
    return [struct.unpack_from("<f", raw, i * 4)[0] for i in range(128)]


def _mult_lanes(st: IpuState) -> list[float]:
    raw = st.regfile.raw("mult_res")
    return [struct.unpack_from("<f", raw, i * 4)[0] for i in range(128)]


def _setup_mult_half_on() -> IpuState:
    """MULT.RC.VV producing 2*3=6 in lanes 0:64 and 2*0=0 in lanes 64:127."""
    st = _new_state()
    r0 = struct.pack("<128f", *([2.0] * 128))
    # RC lanes 0:64 = 3.0, 64:127 = 0.0 -> MULT.RC.VV product 6.0 / 0.0 per lane.
    rc = struct.pack("<128f", *([3.0] * 64 + [0.0] * 64))
    st.xmem.write_address(0x1000, r0)
    st.xmem.write_address(0x2000, rc)
    st.regfile.set_cr(6, 0x1000)
    st.regfile.set_cr(7, 0x2000)
    st.regfile.set_cr(8, 0)
    return st


# ----------------------------------------------------------------------------
# Item 1: masked-zero lanes are exactly 0 entering ACC
# ----------------------------------------------------------------------------
def test_item1_mult_upper_lanes_zero_entering_acc() -> None:
    st = _setup_mult_half_on()
    _load_run(
        st,
        """\
SET lr0 cr6;;
SET lr1 cr7;;
SET lr2 cr8;;
LDR_MULT_REG r0 lr0 cr0;;
LDR_CYCLIC_MULT_REG lr1 cr0 lr2;;
MULT.RC.VV lr2 r0 0 lr2;;
BKPT;;
""",
    )
    m = _mult_lanes(st)
    assert all(v == 6.0 for v in m[0:64]), "lanes 0:64 should be 6.0"
    assert all(v == 0.0 for v in m[64:128]), "lanes 64:127 must be exactly 0"


# ----------------------------------------------------------------------------
# Item 2a: ACC.FIRST writes all 128 lanes -> clobbers upper lanes to 0
# ----------------------------------------------------------------------------
def test_item2_acc_first_clobbers_upper_lanes() -> None:
    st = _setup_mult_half_on()
    # Park sentinel in ALL r_acc lanes beforehand.
    st.regfile.set_r_acc_bytes(struct.pack("<128f", *([SENTINEL] * 128)))
    _load_run(
        st,
        """\
SET lr0 cr6;;
SET lr1 cr7;;
SET lr2 cr8;;
LDR_MULT_REG r0 lr0 cr0;;
LDR_CYCLIC_MULT_REG lr1 cr0 lr2;;
MULT.RC.VV lr2 r0 0 lr2;;
acc.first;;
BKPT;;
""",
    )
    a = _acc_lanes(st)
    assert all(v == 6.0 for v in a[0:64]), "lanes 0:64 take mult result"
    assert all(v == 0.0 for v in a[64:128]), "ACC.FIRST clobbers upper lanes to 0"


# ----------------------------------------------------------------------------
# Item 2b: plain ACC adds 0 to masked lanes -> prior R_ACC[64:127] intact
# ----------------------------------------------------------------------------
def test_item2_plain_acc_preserves_upper_lanes() -> None:
    st = _setup_mult_half_on()
    st.regfile.set_r_acc_bytes(struct.pack("<128f", *([SENTINEL] * 128)))
    _load_run(
        st,
        """\
SET lr0 cr6;;
SET lr1 cr7;;
SET lr2 cr8;;
LDR_MULT_REG r0 lr0 cr0;;
LDR_CYCLIC_MULT_REG lr1 cr0 lr2;;
MULT.RC.VV lr2 r0 0 lr2;;
acc;;
BKPT;;
""",
    )
    a = _acc_lanes(st)
    assert all(v == SENTINEL + 6.0 for v in a[0:64]), "lanes 0:64 accumulate"
    assert all(v == SENTINEL for v in a[64:128]), "plain ACC adds 0, upper intact"


# ----------------------------------------------------------------------------
# Item 3: ACC.STRIDE (h=off, v=off, N=64, offset->start=0) writes [0:64] only
# ----------------------------------------------------------------------------
def test_item3_acc_stride_partial_write_preserves_upper() -> None:
    st = _setup_mult_half_on()
    st.regfile.set_r_acc_bytes(struct.pack("<128f", *([SENTINEL] * 128)))
    st.regfile.set_lr(3, 0)  # offset LR -> (0 % 4) * 32 = start 0
    # elements_in_row=64 -> elements_per_row from lookup; h=off v=off
    _load_run(
        st,
        """\
SET lr0 cr6;;
SET lr1 cr7;;
SET lr2 cr8;;
LDR_MULT_REG r0 lr0 cr0;;
LDR_CYCLIC_MULT_REG lr1 cr0 lr2;;
MULT.RC.VV lr2 r0 0 lr2;;
acc.stride 64 off off lr3;;
BKPT;;
""",
    )
    a = _acc_lanes(st)
    # OBSERVED BEHAVIOR: with h=off v=off, execute_acc_stride sets
    # out_indices = range(128) and base = 0, so it writes ALL 128 lanes
    # (acc[i] = mult[i]). It does NOT leave R_ACC[64:127] unchanged.
    # The upper lanes are overwritten with mult[64:127] (here 0.0).
    assert all(v == 6.0 for v in a[0:64]), "lanes 0:64 <- mult[0:64]"
    assert all(v == 0.0 for v in a[64:128]), (
        "ACC.STRIDE h=off/v=off OVERWRITES lanes 64:127 with mult[64:127] "
        "(== 0 here, NOT the SENTINEL) — it is a full 128-lane copy"
    )


# ----------------------------------------------------------------------------
# Item 4: AGG.SUM.FIRST dest=70, full_xmem_row=0, valid_elements=64
# Post-fix: AGG reduces MULT_RES (not R_ACC). With a half-on MULT (6.0 in lanes
# 0:64, 0.0 above) and valid_elements=64, the reduced sum = 64*6 = 384, written
# to R_ACC[70]. Every OTHER R_ACC lane (pre-parked SENTINEL) is untouched: this
# is the collision-free property -- AGG-into-slot does not disturb stashed lanes.
# ----------------------------------------------------------------------------
def test_item4_agg_sum_first_dest70_valid64() -> None:
    st = _setup_mult_half_on()
    # Pre-park SENTINEL across all R_ACC lanes; AGG must leave all but dest alone.
    st.regfile.set_r_acc_bytes(struct.pack("<128f", *([SENTINEL] * 128)))
    st.set_cr_dstructure(64)  # valid_elements = 64 -> reduce mult_res lanes 0:64
    st.regfile.set_lr(0, 70)  # dest_slot LR value = 70
    _load_run(
        st,
        """\
SET lr1 cr6;;
SET lr5 cr7;;
SET lr2 cr8;;
LDR_MULT_REG r0 lr1 cr0;;
LDR_CYCLIC_MULT_REG lr5 cr0 lr2;;
MULT.RC.VV lr2 r0 0 lr2; AGG.SUM.FIRST lr0 0;;
BKPT;;
""",
    )
    a = _acc_lanes(st)
    assert a[70] == 384.0, f"dest lane 70 should be sum(mult_res[0:64])=384.0, got {a[70]}"
    # Every other lane still SENTINEL -> AGG-into-slot is collision-free.
    for i in range(128):
        if i == 70:
            continue
        assert a[i] == SENTINEL, f"lane {i} changed (collision!)"
