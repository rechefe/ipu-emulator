"""STR_ACC_REG store-width / addressing / slot-overlap verification.

Answers:
1. Can STR_ACC_REG write fewer than the full 512B R_ACC row (single lane /
   N-lane window) via full_xmem_row=0+valid_elements or an offset?
2. Is acc_store a separate issue slot from MULT/ACC (can they co-issue)?

Wide-vector FP32 mode is used so R_ACC lane values are unambiguous floats.
"""

from __future__ import annotations

import struct
import warnings

from ipu_emu.emulator import load_program, run_until_complete
from ipu_emu.execute import decode_instruction_word
from ipu_emu.ipu_math import DType
from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_as.lark_tree import assemble


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
    # STR_ACC_REG emits a [DEBUG ONLY] warning by design — silence it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_until_complete(st)


# ----------------------------------------------------------------------------
# Item 1: store width / addressability
# ----------------------------------------------------------------------------
def test_str_acc_reg_always_writes_full_512b() -> None:
    """STR_ACC_REG writes the entire 512B R_ACC; no lane/window sub-store exists."""
    st = _new_state()
    # Distinct value per lane so a partial store would be visible.
    acc = bytearray(512)
    for i in range(128):
        struct.pack_into("<f", acc, i * 4, float(i + 1))
    st.regfile.set_r_acc_bytes(acc)

    # Pre-fill the destination XMEM region with a sentinel so we can see
    # exactly how many bytes the store overwrites.
    dest = 0x4000
    st.xmem.write_address(dest, bytes([0xEE] * 1024))

    # CR0/CR1 are read-only constants (0 and 1); use a writable CR (CR9) for base.
    st.regfile.set_cr(9, dest)  # base
    st.regfile.set_lr(0, 0)     # offset = 0
    _load_run(st, "STR_ACC_REG lr0 cr9;;\nBKPT;;\n")

    out = st.xmem.read_address(dest, 1024)
    # First 512B == R_ACC verbatim.
    assert out[:512] == bytes(acc), "first 512B must equal R_ACC"
    # Byte 512 onward untouched (sentinel) — store width is exactly 512B.
    assert all(b == 0xEE for b in out[512:1024]), "store must not exceed 512B"


def test_str_acc_reg_no_valid_elements_or_full_xmem_row_operand() -> None:
    """The instruction has only (offset, base) — no width-controlling operand.

    valid_elements (CR15) does NOT narrow the store: setting it to 1 still
    writes all 512 bytes.
    """
    st = _new_state()
    acc = bytearray(512)
    for i in range(128):
        struct.pack_into("<f", acc, i * 4, float(i + 1))
    st.regfile.set_r_acc_bytes(acc)
    st.set_cr_dstructure(1)  # valid_elements = 1 — should have NO effect on store

    dest = 0x4000
    st.xmem.write_address(dest, bytes([0xEE] * 512))
    st.regfile.set_cr(9, dest)
    st.regfile.set_lr(0, 0)
    _load_run(st, "STR_ACC_REG lr0 cr9;;\nBKPT;;\n")

    out = st.xmem.read_address(dest, 512)
    assert out == bytes(acc), "valid_elements=1 must NOT narrow the store"


def test_str_acc_reg_arbitrary_byte_address() -> None:
    """addr = offset + base is byte-granular and arbitrary (not 512B-aligned)."""
    st = _new_state()
    acc = bytearray(512)
    struct.pack_into("<f", acc, 0, 42.0)
    st.regfile.set_r_acc_bytes(acc)

    base = 0x5000
    off = 0x123  # deliberately non-aligned byte offset
    st.regfile.set_cr(9, base)  # CR1 is a read-only constant; use writable CR9
    st.regfile.set_lr(0, off)
    _load_run(st, "STR_ACC_REG lr0 cr9;;\nBKPT;;\n")

    out = st.xmem.read_address(base + off, 4)
    assert struct.unpack_from("<f", out, 0)[0] == 42.0, "stored at offset+base"
    # And nothing landed at the aligned base (proves the offset took effect).
    assert st.xmem.read_address(base, 4) == bytes(4)


# ----------------------------------------------------------------------------
# Item 2: acc_store is a separate slot — co-issues with MULT/ACC
# ----------------------------------------------------------------------------
def test_store_coissues_with_mult_and_acc_same_bundle() -> None:
    """MULT.RC.VV + ACC + STR_ACC_REG in ONE bundle all execute in one cycle.

    Dispatch order is mult -> acc -> ... -> acc_store, and STR_ACC_REG reads
    LIVE R_ACC, so the store captures the value produced by the ACC in the
    SAME bundle.
    """
    st = _new_state()
    r0 = struct.pack("<128f", *([2.0] * 128))
    rc = struct.pack("<128f", *([3.0] * 128))
    st.xmem.write_address(0x1000, r0)
    st.xmem.write_address(0x2000, rc)
    st.regfile.set_cr(6, 0x1000)
    st.regfile.set_cr(7, 0x2000)
    st.regfile.set_cr(8, 0)
    dest = 0x6000
    st.regfile.set_cr(9, dest)  # CR9 (writable) = store base address
    # Pre-seed R_ACC with 10.0 so ACC (not ACC.FIRST) gives 10+6=16.
    st.regfile.set_r_acc_bytes(struct.pack("<128f", *([10.0] * 128)))

    # Single compound bundle: load operands first (separate bundles), then the
    # co-issue bundle carrying MULT (mult slot) + ACC (acc slot) + STR_ACC_REG
    # (acc_store slot) together.
    asm = """\
SET lr0 cr6;;
SET lr1 cr7;;
SET lr2 cr8;;
SET lr3 cr0;;
LDR_MULT_REG r0 lr0 cr0;;
LDR_CYCLIC_MULT_REG lr1 cr0 lr2;;
MULT.RC.VV lr2 r0 0 lr2; ACC; STR_ACC_REG lr3 cr9;;
BKPT;;
"""
    _load_run(st, asm)

    # Live R_ACC after the bundle: 10 + (2*3) = 16.0 on every lane.
    raw = st.regfile.raw("r_acc")
    assert struct.unpack_from("<f", raw, 0)[0] == 16.0

    # The store in the SAME bundle captured the post-ACC value (16.0),
    # proving acc_store co-issued and ran after acc in the dispatch order.
    out = st.xmem.read_address(dest, 512)
    for i in range(128):
        assert struct.unpack_from("<f", out, i * 4)[0] == 16.0, f"lane {i}"
