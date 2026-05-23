"""Wide-vector debug mode (GitHub issue #33).

XMEM transfer sizes stay architectural (128-byte loads for r in normal mode);
in wide-vector mode ``LDR_MULT_REG`` / ``LDR_CYCLIC_MULT_REG`` consume 512 bytes
per transfer as 128×32-bit lanes. LR/CR are unchanged.
"""

from __future__ import annotations

import struct

import pytest

from ipu_emu.emulator import load_program, run_until_complete
from ipu_emu.execute import decode_instruction_word
from ipu_emu.ipu import EmulatorError
from ipu_emu.ipu_math import DType
from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_as.lark_tree import assemble


def _run_wide(
    asm: str,
    *,
    arithmetic: WideVectorArithmetic = WideVectorArithmetic.FP32,
    quantize_output: bool = False,
    cr: dict[int, int] | None = None,
) -> IpuState:
    encoded = assemble(asm)
    decoded = [decode_instruction_word(w) for w in encoded]
    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=arithmetic,
        wide_vector_quantize_output=quantize_output,
    )
    state.regfile.set_cr(15, DType.INT8)
    if cr:
        for idx, val in cr.items():
            state.regfile.set_cr(idx, val)
    load_program(state, decoded)
    run_until_complete(state)
    return state


class TestWideVectorFp32:
    def test_mult_ee_fp32_no_quantization_in_acc(self) -> None:
        """128 float lanes multiply element-wise; acc holds float products."""
        r0 = struct.pack("<128f", *([2.0] * 128))
        rc = struct.pack("<128f", *([3.0] * 128))
        state = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
        state.regfile.set_cr(15, DType.INT8)
        state.xmem.write_address(0x1000, r0)
        state.xmem.write_address(0x2000, rc)
        asm = """\
SET lr0 cr6;;
SET lr1 cr7;;
SET lr2 cr8;;
LDR_MULT_REG r0 lr0 cr0;;
LDR_CYCLIC_MULT_REG lr1 cr0 lr2;;
RESET_ACC;;
MULT.EE r0 lr2 0 lr2;;
acc.first;;
BKPT;;
"""
        state.regfile.set_cr(6, 0x1000)
        state.regfile.set_cr(7, 0x2000)
        state.regfile.set_cr(8, 0)
        encoded = assemble(asm)
        load_program(state, [decode_instruction_word(w) for w in encoded])
        run_until_complete(state)
        acc = state.regfile.raw("r_acc")
        for i in range(128):
            v = struct.unpack_from("<f", acc, i * 4)[0]
            assert v == pytest.approx(6.0), f"lane {i}"

    def test_aaq_noop_unless_quantize_flag(self) -> None:
        state = _run_wide(
            """\
SET lr0 cr6;;
SET lr1 cr7;;
SET lr2 cr8;;
LDR_MULT_REG r0 lr0 cr0;;
LDR_CYCLIC_MULT_REG lr1 cr0 lr2;;
RESET_ACC;;
MULT.EE r0 lr2 0 lr2;;
acc.first;;
aaq;;
BKPT;;
""",
            cr={6: 0x1000, 7: 0x2000, 8: 0},
        )
        assert state.regfile.get_post_aaq_reg() == bytearray(512)

    def test_aaq_quantize_fp32_acc_for_comparison(self) -> None:
        # Use exact float product 3.0 so rounding is unambiguous (round-half-even).
        r0 = struct.pack("<128f", *([1.5] * 128))
        rc = struct.pack("<128f", *([2.0] * 128))
        state = IpuState(
            wide_vector_debug=True,
            wide_vector_arithmetic=WideVectorArithmetic.FP32,
            wide_vector_quantize_output=True,
        )
        state.regfile.set_cr(15, DType.INT8)
        state.xmem.write_address(0x1000, r0)
        state.xmem.write_address(0x2000, rc)
        asm = """\
SET lr0 cr6;;
SET lr1 cr7;;
SET lr2 cr8;;
LDR_MULT_REG r0 lr0 cr0;;
LDR_CYCLIC_MULT_REG lr1 cr0 lr2;;
RESET_ACC;;
MULT.EE r0 lr2 0 lr2;;
acc.first;;
aaq;;
BKPT;;
"""
        state.regfile.set_cr(6, 0x1000)
        state.regfile.set_cr(7, 0x2000)
        state.regfile.set_cr(8, 0)
        encoded = assemble(asm)
        load_program(state, [decode_instruction_word(w) for w in encoded])
        run_until_complete(state)
        out = state.regfile.get_post_aaq_reg()
        assert all(b == 3 for b in out[:128])
        assert out[128:] == bytearray(384)


class TestWideVectorInt32:
    def test_mult_ee_int32_wrap_multiply(self) -> None:
        r0 = struct.pack("<128i", *([70000] * 128))
        rc = struct.pack("<128i", *([70000] * 128))
        state = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.INT32)
        state.regfile.set_cr(15, DType.INT8)
        state.xmem.write_address(0x1000, r0)
        state.xmem.write_address(0x2000, rc)
        asm = """\
SET lr0 cr6;;
SET lr1 cr7;;
SET lr2 cr8;;
LDR_MULT_REG r0 lr0 cr0;;
LDR_CYCLIC_MULT_REG lr1 cr0 lr2;;
RESET_ACC;;
MULT.EE r0 lr2 0 lr2;;
acc.first;;
BKPT;;
"""
        state.regfile.set_cr(6, 0x1000)
        state.regfile.set_cr(7, 0x2000)
        state.regfile.set_cr(8, 0)
        encoded = assemble(asm)
        load_program(state, [decode_instruction_word(w) for w in encoded])
        run_until_complete(state)
        acc = state.regfile.raw("r_acc")
        expected = (70000 * 70000) & 0xFFFFFFFF
        if expected >= 0x80000000:
            expected -= 0x100000000
        for i in range(128):
            v = struct.unpack_from("<i", acc, i * 4)[0]
            assert v == expected, f"lane {i}"


class TestWideVectorR0R1Isolation:
    """Encoding 0 (R0) and 1 (R1) must not share debug staging storage."""

    def test_r0_and_r1_independent(self) -> None:
        r0 = struct.pack("<128f", *([1.0] * 128))
        r1 = struct.pack("<128f", *([99.0] * 128))
        rc = struct.pack("<128f", *([2.0] * 128))

        def run_mult(which: str) -> float:
            st = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
            st.regfile.set_cr(15, DType.INT8)
            st.xmem.write_address(0x1000, r0)
            st.xmem.write_address(0x1100, r1)
            st.xmem.write_address(0x2000, rc)
            st.regfile.set_cr(6, 0x1000)
            st.regfile.set_cr(7, 0x1100)
            st.regfile.set_cr(8, 0x2000)
            st.regfile.set_cr(9, 0)
            asm = f"""\
SET lr0 cr6;;
SET lr1 cr7;;
SET lr2 cr8;;
SET lr3 cr9;;
LDR_MULT_REG r0 lr0 cr0;;
LDR_MULT_REG r1 lr1 cr0;;
LDR_CYCLIC_MULT_REG lr2 cr0 lr3;;
RESET_ACC;;
MULT.EE {which} lr3 0 lr3;;
acc.first;;
BKPT;;
"""
            encoded = assemble(asm)
            load_program(st, [decode_instruction_word(w) for w in encoded])
            run_until_complete(st)
            return struct.unpack_from("<f", st.regfile.raw("r_acc"), 0)[0]

        assert run_mult("r0") == pytest.approx(2.0)
        assert run_mult("r1") == pytest.approx(198.0)


class TestWideVectorCyclicIndex:
    """``LDR_CYCLIC_MULT_REG`` must honour ``index`` in wide mode (512-byte chunk)."""

    def test_ldr_cyclic_nonzero_index_fp32(self) -> None:
        zeros = struct.pack("<128f", *([0.0] * 128))
        nines = struct.pack("<128f", *([9.0] * 128))
        twos = struct.pack("<128f", *([2.0] * 128))
        st = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
        st.regfile.set_cr(15, DType.INT8)
        st.xmem.write_address(0x2000, zeros)
        st.xmem.write_address(0x2200, nines)
        st.xmem.write_address(0x1000, twos)
        st.regfile.set_cr(6, 0x2000)
        st.regfile.set_cr(7, 0)
        st.regfile.set_cr(8, 0x2200)
        st.regfile.set_cr(9, 512)
        st.regfile.set_cr(10, 0x1000)
        st.regfile.set_cr(11, 512)
        asm = """\
SET lr0 cr6;;
SET lr1 cr7;;
LDR_CYCLIC_MULT_REG lr0 cr0 lr1;;
SET lr2 cr8;;
SET lr3 cr9;;
LDR_CYCLIC_MULT_REG lr2 cr0 lr3;;
SET lr4 cr10;;
LDR_MULT_REG r0 lr4 cr0;;
SET lr5 cr11;;
RESET_ACC;;
MULT.EE r0 lr5 0 lr5;;
acc.first;;
BKPT;;
"""
        encoded = assemble(asm)
        load_program(st, [decode_instruction_word(w) for w in encoded])
        run_until_complete(st)
        acc = st.regfile.raw("r_acc")
        for i in range(128):
            assert struct.unpack_from("<f", acc, i * 4)[0] == pytest.approx(18.0), f"lane {i}"


class TestWideVectorPadding:
    """Lanes past the r_cyclic byte window use padding (×1) in wide FP32 mode."""

    def test_mult_ve_cr_fp32_boundary_padding(self) -> None:
        # cyclic_offset=384 (aligned): lane 31 ends at byte 508; lane 32 starts at 512 → pad.
        buf = bytearray(512)
        for k in range(32):
            struct.pack_into("<f", buf, 384 + k * 4, 3.0)
        st = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
        st.regfile.set_cr(15, DType.INT8)
        st.regfile.set_cr(1, 2)  # low byte 2 → scalar 2.0 in wide FP32 path
        st.regfile.set_r_cyclic_at(0, buf)
        st.regfile.set_cr(6, 384)
        st.regfile.set_cr(7, 0)
        asm = """\
SET lr0 cr6;;
SET lr2 cr7;;
MULT.VE.CR lr0 0 lr2 cr1;;
RESET_ACC;;
acc.first;;
BKPT;;
"""
        encoded = assemble(asm)
        load_program(st, [decode_instruction_word(w) for w in encoded])
        run_until_complete(st)
        mult_res = st.regfile.raw("mult_res")
        for i in range(32):
            assert struct.unpack_from("<f", mult_res, i * 4)[0] == pytest.approx(6.0), f"lane {i}"
        for i in range(32, 128):
            assert struct.unpack_from("<f", mult_res, i * 4)[0] == pytest.approx(2.0), f"lane {i}"

    def test_mult_ve_fp32_wide_cyclic_past_boundary(self) -> None:
        """MULT.VE.CYCLIC (wide FP32): RC lanes wrap at 512 bytes."""
        buf = bytearray(512)
        for k in range(32):
            struct.pack_into("<f", buf, 384 + k * 4, 3.0)
        for k in range(96):
            struct.pack_into("<f", buf, k * 4, 5.0)
        st = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
        st.regfile.set_cr(15, DType.INT8)
        st.regfile.set_cr(0, 0)
        st.regfile.set_r_cyclic_at(0, buf)
        st.xmem.write_address(0x1000, struct.pack("<128f", *([2.0] * 128)))
        st.regfile.set_cr(10, 0x1000)
        st.regfile.set_cr(6, 384)
        st.regfile.set_cr(7, 0)
        st.regfile.set_cr(8, 0)
        asm = """\
SET lr4 cr10;;
LDR_MULT_REG r0 lr4 cr0;;
SET lr0 cr6;;
SET lr2 cr7;;
SET lr3 cr8;;
MULT.VE.CYCLIC lr0 0 lr2 lr3;;
RESET_ACC;;
acc.first;;
BKPT;;
"""
        encoded = assemble(asm)
        load_program(st, [decode_instruction_word(w) for w in encoded])
        run_until_complete(st)
        mult_res = st.regfile.raw("mult_res")
        for i in range(32):
            assert struct.unpack_from("<f", mult_res, i * 4)[0] == pytest.approx(6.0), f"lane {i}"
        for i in range(32, 128):
            assert struct.unpack_from("<f", mult_res, i * 4)[0] == pytest.approx(10.0), f"lane {i}"

    def test_mult_ve_fp32_wide_padded_past_boundary(self) -> None:
        """MULT.VE.PADDED (wide FP32): lanes past byte 511 use ×1."""
        buf = bytearray(512)
        for k in range(32):
            struct.pack_into("<f", buf, 384 + k * 4, 3.0)
        for k in range(96):
            struct.pack_into("<f", buf, k * 4, 5.0)
        st = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
        st.regfile.set_cr(15, DType.INT8)
        st.regfile.set_cr(0, 0)
        st.regfile.set_r_cyclic_at(0, buf)
        st.xmem.write_address(0x1000, struct.pack("<128f", *([2.0] * 128)))
        st.regfile.set_cr(10, 0x1000)
        st.regfile.set_cr(6, 384)
        st.regfile.set_cr(7, 0)
        st.regfile.set_cr(8, 0)
        asm = """\
SET lr4 cr10;;
LDR_MULT_REG r0 lr4 cr0;;
SET lr0 cr6;;
SET lr2 cr7;;
SET lr3 cr8;;
MULT.VE.PADDED lr0 0 lr2 lr3;;
RESET_ACC;;
acc.first;;
BKPT;;
"""
        encoded = assemble(asm)
        load_program(st, [decode_instruction_word(w) for w in encoded])
        run_until_complete(st)
        mult_res = st.regfile.raw("mult_res")
        for i in range(32):
            assert struct.unpack_from("<f", mult_res, i * 4)[0] == pytest.approx(6.0), f"lane {i}"
        for i in range(32, 128):
            assert struct.unpack_from("<f", mult_res, i * 4)[0] == pytest.approx(2.0), f"lane {i}"


class TestWideVectorAggInt32:
    def test_agg_sum_inv_int32_wide(self) -> None:
        """agg sum inv with INT32 wide lanes: 128×4 = 512 → inv rounds to int32 bits."""
        acc = bytearray(512)
        struct.pack_into("<128i", acc, 0, *([4] * 128))
        st = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.INT32)
        st.regfile.set_cr(15, DType.INT8)
        st.regfile.set_r_acc_bytes(acc)
        st.regfile.set_lr(1, 128)
        asm = """\
agg sum inv lr1 cr0 aaq0;;
BKPT;;
"""
        encoded = assemble(asm)
        load_program(st, [decode_instruction_word(w) for w in encoded])
        run_until_complete(st)
        assert st.regfile.get_aaq(0) == 0  # 1/512 rounds to 0 as int32


class TestWideVectorAlignment:
    def test_misaligned_cyclic_offset_raises(self) -> None:
        st = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
        st.regfile.set_cr(15, DType.INT8)
        st.xmem.write_address(0x1000, struct.pack("<128f", *([1.0] * 128)))
        st.xmem.write_address(0x2000, struct.pack("<128f", *([2.0] * 128)))
        st.regfile.set_cr(6, 0x1000)
        st.regfile.set_cr(7, 0x2000)
        st.regfile.set_cr(8, 0)
        st.regfile.set_cr(9, 1)
        asm = """\
SET lr0 cr6;;
SET lr1 cr7;;
SET lr2 cr8;;
LDR_MULT_REG r0 lr0 cr0;;
LDR_CYCLIC_MULT_REG lr1 cr0 lr2;;
SET lr3 cr9;;
RESET_ACC;;
MULT.EE r0 lr3 0 lr3;;
BKPT;;
"""
        encoded = assemble(asm)
        load_program(st, [decode_instruction_word(w) for w in encoded])
        with pytest.raises(EmulatorError, match="4-byte aligned"):
            run_until_complete(st)
