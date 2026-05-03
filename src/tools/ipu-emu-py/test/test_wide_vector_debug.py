"""Wide-vector debug mode (GitHub issue #33).

XMEM transfer sizes stay architectural (128-byte loads for r in normal mode);
in wide-vector mode ``ldr_mult_reg`` / ``ldr_cyclic_mult_reg`` consume 512 bytes
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
) -> IpuState:
    encoded = assemble(asm)
    decoded = [decode_instruction_word(w) for w in encoded]
    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=arithmetic,
        wide_vector_quantize_output=quantize_output,
    )
    state.regfile.set_cr(15, DType.INT8)
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
set lr0 0x1000;;
set lr1 0x2000;;
set lr2 0;;
ldr_mult_reg r0 lr0 cr0;;
ldr_cyclic_mult_reg lr1 cr0 lr2;;
reset_acc;;
mult.ee r0 lr2 lr2 lr2;;
acc.first;;
bkpt;;
"""
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
set lr0 0x1000;;
set lr1 0x2000;;
set lr2 0;;
ldr_mult_reg r0 lr0 cr0;;
ldr_cyclic_mult_reg lr1 cr0 lr2;;
reset_acc;;
mult.ee r0 lr2 lr2 lr2;;
acc.first;;
aaq;;
bkpt;;
""",
        )
        assert state.regfile.get_aaq_result() == bytearray(128)

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
set lr0 0x1000;;
set lr1 0x2000;;
set lr2 0;;
ldr_mult_reg r0 lr0 cr0;;
ldr_cyclic_mult_reg lr1 cr0 lr2;;
reset_acc;;
mult.ee r0 lr2 lr2 lr2;;
acc.first;;
aaq;;
bkpt;;
"""
        encoded = assemble(asm)
        load_program(state, [decode_instruction_word(w) for w in encoded])
        run_until_complete(state)
        out = state.regfile.get_aaq_result()
        assert all(b == 3 for b in out)


class TestWideVectorInt32:
    def test_mult_ee_int32_wrap_multiply(self) -> None:
        r0 = struct.pack("<128i", *([70000] * 128))
        rc = struct.pack("<128i", *([70000] * 128))
        state = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.INT32)
        state.regfile.set_cr(15, DType.INT8)
        state.xmem.write_address(0x1000, r0)
        state.xmem.write_address(0x2000, rc)
        asm = """\
set lr0 0x1000;;
set lr1 0x2000;;
set lr2 0;;
ldr_mult_reg r0 lr0 cr0;;
ldr_cyclic_mult_reg lr1 cr0 lr2;;
reset_acc;;
mult.ee r0 lr2 lr2 lr2;;
acc.first;;
bkpt;;
"""
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


class TestWideVectorMemBypassIsolation:
    """Encoding 0 (r0) and 2 (mem_bypass) must not share debug staging storage."""

    def test_r0_and_mem_bypass_independent(self) -> None:
        r0 = struct.pack("<128f", *([1.0] * 128))
        mb = struct.pack("<128f", *([99.0] * 128))
        rc = struct.pack("<128f", *([2.0] * 128))

        def run_mult(which: str) -> float:
            st = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
            st.regfile.set_cr(15, DType.INT8)
            st.xmem.write_address(0x1000, r0)
            st.xmem.write_address(0x1100, mb)
            st.xmem.write_address(0x2000, rc)
            asm = f"""\
set lr0 0x1000;;
set lr1 0x1100;;
set lr2 0x2000;;
set lr3 0;;
ldr_mult_reg r0 lr0 cr0;;
ldr_mult_reg mem_bypass lr1 cr0;;
ldr_cyclic_mult_reg lr2 cr0 lr3;;
reset_acc;;
mult.ee {which} lr3 lr3 lr3;;
acc.first;;
bkpt;;
"""
            encoded = assemble(asm)
            load_program(st, [decode_instruction_word(w) for w in encoded])
            run_until_complete(st)
            return struct.unpack_from("<f", st.regfile.raw("r_acc"), 0)[0]

        assert run_mult("r0") == pytest.approx(2.0)
        assert run_mult("mem_bypass") == pytest.approx(198.0)


class TestWideVectorCyclicIndex:
    """``ldr_cyclic_mult_reg`` must honour ``index`` in wide mode (512-byte chunk)."""

    def test_ldr_cyclic_nonzero_index_fp32(self) -> None:
        zeros = struct.pack("<128f", *([0.0] * 128))
        nines = struct.pack("<128f", *([9.0] * 128))
        twos = struct.pack("<128f", *([2.0] * 128))
        st = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
        st.regfile.set_cr(15, DType.INT8)
        st.xmem.write_address(0x2000, zeros)
        st.xmem.write_address(0x2200, nines)
        st.xmem.write_address(0x1000, twos)
        asm = """\
set lr0 0x2000;;
set lr1 0;;
ldr_cyclic_mult_reg lr0 cr0 lr1;;
set lr2 0x2200;;
set lr3 512;;
ldr_cyclic_mult_reg lr2 cr0 lr3;;
set lr4 0x1000;;
ldr_mult_reg r0 lr4 cr0;;
set lr5 512;;
reset_acc;;
mult.ee r0 lr5 lr5 lr5;;
acc.first;;
bkpt;;
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
        asm = """\
set lr0 384;;
set lr1 0;;
set lr2 0;;
mult.ve.cr lr0 lr1 lr2 cr1;;
reset_acc;;
acc.first;;
bkpt;;
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
        asm = """\
agg sum inv cr0 aaq0;;
bkpt;;
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
        asm = """\
set lr0 0x1000;;
set lr1 0x2000;;
set lr2 0;;
ldr_mult_reg r0 lr0 cr0;;
ldr_cyclic_mult_reg lr1 cr0 lr2;;
set lr3 1;;
reset_acc;;
mult.ee r0 lr3 lr3 lr3;;
bkpt;;
"""
        encoded = assemble(asm)
        load_program(st, [decode_instruction_word(w) for w in encoded])
        with pytest.raises(EmulatorError, match="4-byte aligned"):
            run_until_complete(st)
