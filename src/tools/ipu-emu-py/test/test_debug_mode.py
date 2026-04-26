"""Tests for the 32-bit FP32 debug mode (IpuState.debug_32bit=True).

In debug mode every formerly-8-bit input register (r0, r1, r_cyclic) is
widened to 128 × float32.  ldr_mult_reg loads 512 bytes (128 × float32)
from XMEM; ldr_cyclic_mult_reg overwrites the full 512-byte cyclic buffer
in one call.  The accumulator (r_acc) stays 128 × float32 — no format
change — but is now fed by float32 products instead of INT8/FP8 products.
"""

from __future__ import annotations

import struct

import pytest

from ipu_emu.execute import decode_instruction_word
from ipu_emu.emulator import load_program, run_until_complete
from ipu_emu.ipu_state import IpuState

from ipu_as.lark_tree import assemble


# ---------------------------------------------------------------------------
# Helpers (mirrors test_execute.py)
# ---------------------------------------------------------------------------


def _make_state(asm_code: str, debug_32bit: bool = False,
                debug_quantize: bool = False) -> IpuState:
    encoded = assemble(asm_code)
    decoded = [decode_instruction_word(w) for w in encoded]
    state = IpuState(debug_32bit=debug_32bit, debug_quantize=debug_quantize)
    load_program(state, decoded)
    return state


def _run(asm_code: str, debug_32bit: bool = False,
         debug_quantize: bool = False) -> IpuState:
    state = _make_state(asm_code, debug_32bit=debug_32bit,
                        debug_quantize=debug_quantize)
    run_until_complete(state)
    return state


# float32 helpers
def _pack_f32x128(val: float) -> bytes:
    """128 identical float32 values packed as 512 bytes."""
    return struct.pack("<128f", *([val] * 128))


def _unpack_f32x128(data: bytes) -> list[float]:
    """Unpack 512 bytes as 128 float32 values."""
    return list(struct.unpack_from("<128f", data))


# ---------------------------------------------------------------------------
# Basic multiply
# ---------------------------------------------------------------------------


class TestDebugBasicMultiply:
    def test_fp32_multiply_ee(self):
        """mult.ee in debug mode: 200.0 × 150.0 = 30000.0 (impossible as INT8)."""
        state = _make_state(
            """\
set lr0 0x1000;;
ldr_mult_reg r0 lr0 cr0;;
set lr1 0x2000;;
set lr2 0;;
ldr_cyclic_mult_reg lr1 cr0 lr2;;
reset_acc;;
mult.ee r0 lr0 lr0 lr0;
acc.first;;
set lr9 0x3000;;
str_acc_reg lr9 cr0;;
bkpt;;
""",
            debug_32bit=True,
        )
        state.xmem.write_address(0x1000, _pack_f32x128(200.0))
        state.xmem.write_address(0x2000, _pack_f32x128(150.0))
        run_until_complete(state)

        acc_bytes = state.xmem.read_address(0x3000, 512)
        values = _unpack_f32x128(acc_bytes)
        for i, v in enumerate(values):
            assert v == pytest.approx(30000.0), f"element {i}: expected 30000.0, got {v}"

    def test_fp32_multiply_avoids_int8_sign_issue(self):
        """Values 200 and 150 are negative in INT8 (−56, −106); in FP32 they are positive."""
        state = _make_state(
            """\
set lr0 0x1000;;
ldr_mult_reg r0 lr0 cr0;;
set lr1 0x2000;;
set lr2 0;;
ldr_cyclic_mult_reg lr1 cr0 lr2;;
reset_acc;;
mult.ee r0 lr0 lr0 lr0;
acc.first;;
set lr9 0x3000;;
str_acc_reg lr9 cr0;;
bkpt;;
""",
            debug_32bit=False,
        )
        state.xmem.write_address(0x1000, bytes([200] * 128))
        state.xmem.write_address(0x2000, bytes([150] * 512))
        run_until_complete(state)

        # Normal mode: INT8 → signed, so 200 = −56, 150 = −106; product = 5936
        acc_bytes_normal = state.xmem.read_address(0x3000, 512)
        normal_vals = list(struct.unpack_from("<128i", acc_bytes_normal))
        assert normal_vals[0] == (-56) * (-106)  # = 5936

        # Debug mode: 200.0 × 150.0 = 30000.0
        state2 = _make_state(
            """\
set lr0 0x1000;;
ldr_mult_reg r0 lr0 cr0;;
set lr1 0x2000;;
set lr2 0;;
ldr_cyclic_mult_reg lr1 cr0 lr2;;
reset_acc;;
mult.ee r0 lr0 lr0 lr0;
acc.first;;
set lr9 0x3000;;
str_acc_reg lr9 cr0;;
bkpt;;
""",
            debug_32bit=True,
        )
        state2.xmem.write_address(0x1000, _pack_f32x128(200.0))
        state2.xmem.write_address(0x2000, _pack_f32x128(150.0))
        run_until_complete(state2)

        acc_bytes_debug = state2.xmem.read_address(0x3000, 512)
        debug_vals = _unpack_f32x128(acc_bytes_debug)
        assert debug_vals[0] == pytest.approx(30000.0)

    def test_fp32_fractional_precision(self):
        """float32 preserves fractional values that INT8 would lose."""
        state = _make_state(
            """\
set lr0 0x1000;;
ldr_mult_reg r0 lr0 cr0;;
set lr1 0x2000;;
set lr2 0;;
ldr_cyclic_mult_reg lr1 cr0 lr2;;
reset_acc;;
mult.ee r0 lr0 lr0 lr0;
acc.first;;
set lr9 0x3000;;
str_acc_reg lr9 cr0;;
bkpt;;
""",
            debug_32bit=True,
        )
        state.xmem.write_address(0x1000, _pack_f32x128(1.5))
        state.xmem.write_address(0x2000, _pack_f32x128(2.0))
        run_until_complete(state)

        acc_bytes = state.xmem.read_address(0x3000, 512)
        values = _unpack_f32x128(acc_bytes)
        for i, v in enumerate(values):
            assert v == pytest.approx(3.0), f"element {i}: expected 3.0, got {v}"


# ---------------------------------------------------------------------------
# Multi-cycle accumulation
# ---------------------------------------------------------------------------


class TestDebugAccumulation:
    def test_accumulate_multiple_cycles(self):
        """Accumulate over 4 cycles: result should be 4 × (a × b)."""
        asm = """\
set lr0 0x1000;;
ldr_mult_reg r0 lr0 cr0;;
set lr1 0x2000;;
set lr2 0;;
ldr_cyclic_mult_reg lr1 cr0 lr2;;
reset_acc;;
mult.ee r0 lr0 lr0 lr0;
acc.first;;
mult.ee r0 lr0 lr0 lr0;
acc;;
mult.ee r0 lr0 lr0 lr0;
acc;;
mult.ee r0 lr0 lr0 lr0;
acc;;
set lr9 0x3000;;
str_acc_reg lr9 cr0;;
bkpt;;
"""
        state = _make_state(asm, debug_32bit=True)
        state.xmem.write_address(0x1000, _pack_f32x128(3.0))
        state.xmem.write_address(0x2000, _pack_f32x128(4.0))
        run_until_complete(state)

        acc_bytes = state.xmem.read_address(0x3000, 512)
        values = _unpack_f32x128(acc_bytes)
        for i, v in enumerate(values):
            assert v == pytest.approx(48.0), f"element {i}: expected 48.0, got {v}"

    def test_large_values_no_overflow(self):
        """Values that would overflow INT32 accumulation are fine in FP32."""
        # 1000.0 × 1000.0 = 1e6; accumulated 128 times would overflow INT32 (max ~2.1e9)
        # but in FP32 accumulation it stays exact.
        asm = """\
set lr0 0x1000;;
ldr_mult_reg r0 lr0 cr0;;
set lr1 0x2000;;
set lr2 0;;
ldr_cyclic_mult_reg lr1 cr0 lr2;;
reset_acc;;
mult.ee r0 lr0 lr0 lr0;
acc.first;;
set lr9 0x3000;;
str_acc_reg lr9 cr0;;
bkpt;;
"""
        state = _make_state(asm, debug_32bit=True)
        state.xmem.write_address(0x1000, _pack_f32x128(1000.0))
        state.xmem.write_address(0x2000, _pack_f32x128(1000.0))
        run_until_complete(state)

        acc_bytes = state.xmem.read_address(0x3000, 512)
        values = _unpack_f32x128(acc_bytes)
        for i, v in enumerate(values):
            assert v == pytest.approx(1_000_000.0), f"element {i}: expected 1e6, got {v}"


# ---------------------------------------------------------------------------
# AAQ / quantize option
# ---------------------------------------------------------------------------


class TestDebugAaq:
    def test_aaq_noop_without_quantize(self):
        """execute_aaq is a no-op in debug mode when debug_quantize=False."""
        asm = """\
set lr0 0x1000;;
ldr_mult_reg r0 lr0 cr0;;
set lr1 0x2000;;
set lr2 0;;
ldr_cyclic_mult_reg lr1 cr0 lr2;;
reset_acc;;
mult.ee r0 lr0 lr0 lr0;
acc.first;;
aaq;;
set lr9 0x3000;;
str_acc_reg lr9 cr0;;
bkpt;;
"""
        state = _make_state(asm, debug_32bit=True, debug_quantize=False)
        state.xmem.write_address(0x1000, _pack_f32x128(5.0))
        state.xmem.write_address(0x2000, _pack_f32x128(6.0))
        run_until_complete(state)

        # aaq is a no-op → r_acc still holds 30.0 for every element
        acc_bytes = state.xmem.read_address(0x3000, 512)
        values = _unpack_f32x128(acc_bytes)
        for i, v in enumerate(values):
            assert v == pytest.approx(30.0), f"element {i}: expected 30.0, got {v}"

    def test_debug_quantize_clamps_to_int8(self):
        """debug_quantize=True clamps float32 acc values to INT8 range [-128, 127]."""
        asm = """\
set lr0 0x1000;;
ldr_mult_reg r0 lr0 cr0;;
set lr1 0x2000;;
set lr2 0;;
ldr_cyclic_mult_reg lr1 cr0 lr2;;
reset_acc;;
mult.ee r0 lr0 lr0 lr0;
acc.first;;
aaq;;
set lr9 0x3000;;
xmem.store_aaq_result lr9 cr0;;
bkpt;;
"""
        state = _make_state(asm, debug_32bit=True, debug_quantize=True)
        state.xmem.write_address(0x1000, _pack_f32x128(5.0))
        state.xmem.write_address(0x2000, _pack_f32x128(6.0))
        run_until_complete(state)

        aaq_bytes = state.xmem.read_address(0x3000, 128)
        # 5.0 × 6.0 = 30.0 → int(round(30.0)) = 30 → fits in INT8
        for i, b in enumerate(aaq_bytes):
            signed = b if b < 128 else b - 256
            assert signed == 30, f"byte {i}: expected 30, got {signed}"

    def test_debug_quantize_out_of_range_clamps(self):
        """Values outside INT8 range are clamped to [-128, 127]."""
        asm = """\
set lr0 0x1000;;
ldr_mult_reg r0 lr0 cr0;;
set lr1 0x2000;;
set lr2 0;;
ldr_cyclic_mult_reg lr1 cr0 lr2;;
reset_acc;;
mult.ee r0 lr0 lr0 lr0;
acc.first;;
aaq;;
set lr9 0x3000;;
xmem.store_aaq_result lr9 cr0;;
bkpt;;
"""
        state = _make_state(asm, debug_32bit=True, debug_quantize=True)
        # 200.0 × 200.0 = 40000.0 > 127 → clamp to 127
        state.xmem.write_address(0x1000, _pack_f32x128(200.0))
        state.xmem.write_address(0x2000, _pack_f32x128(200.0))
        run_until_complete(state)

        aaq_bytes = state.xmem.read_address(0x3000, 128)
        for i, b in enumerate(aaq_bytes):
            signed = b if b < 128 else b - 256
            assert signed == 127, f"byte {i}: expected 127 (clamped), got {signed}"


# ---------------------------------------------------------------------------
# Memory sizes unchanged
# ---------------------------------------------------------------------------


class TestDebugMemorySizes:
    def test_r_cyclic_still_512_bytes(self):
        state = IpuState(debug_32bit=True)
        assert len(state.regfile.get_r_cyclic_at(0, 512)) == 512

    def test_r_acc_still_512_bytes(self):
        state = IpuState(debug_32bit=True)
        assert len(state.regfile.get_r_acc_bytes()) == 512

    def test_aaq_result_still_128_bytes(self):
        state = IpuState(debug_32bit=True)
        assert len(state.regfile.get_aaq_result()) == 128

    def test_str_acc_reg_writes_512_bytes(self):
        """str_acc_reg always stores 512 bytes, carrying float32 data in debug mode."""
        asm = """\
set lr0 0x1000;;
ldr_mult_reg r0 lr0 cr0;;
set lr1 0x2000;;
set lr2 0;;
ldr_cyclic_mult_reg lr1 cr0 lr2;;
reset_acc;;
mult.ee r0 lr0 lr0 lr0;
acc.first;;
set lr9 0x3000;;
str_acc_reg lr9 cr0;;
bkpt;;
"""
        state = _make_state(asm, debug_32bit=True)
        state.xmem.write_address(0x1000, _pack_f32x128(2.0))
        state.xmem.write_address(0x2000, _pack_f32x128(3.0))
        run_until_complete(state)

        acc_bytes = state.xmem.read_address(0x3000, 512)
        assert len(acc_bytes) == 512
        values = _unpack_f32x128(acc_bytes)
        for i, v in enumerate(values):
            assert v == pytest.approx(6.0), f"element {i}: expected 6.0, got {v}"


# ---------------------------------------------------------------------------
# Normal mode unaffected
# ---------------------------------------------------------------------------


class TestDebugModeIsolation:
    def test_normal_mode_unaffected(self):
        """Running with debug_32bit=False produces identical results to before."""
        asm = """\
set lr0 0x1000;;
ldr_mult_reg r0 lr0 cr0;;
set lr1 0x2000;;
set lr2 0;;
ldr_cyclic_mult_reg lr1 cr0 lr2;;
reset_acc;;
mult.ee r0 lr0 lr0 lr0;
acc.first;;
set lr9 0x3000;;
str_acc_reg lr9 cr0;;
bkpt;;
"""
        r0_data = bytes([2] * 128)
        cyclic_data = bytes([3] * 512)

        state = _make_state(asm, debug_32bit=False)
        state.xmem.write_address(0x1000, r0_data)
        state.xmem.write_address(0x2000, cyclic_data)
        run_until_complete(state)

        acc_bytes = state.xmem.read_address(0x3000, 512)
        words = list(struct.unpack_from("<128i", acc_bytes))
        for i, w in enumerate(words):
            assert w == 6, f"element {i}: expected 6, got {w}"

    def test_debug_flag_default_is_false(self):
        state = IpuState()
        assert state.debug_32bit is False
        assert state.debug_quantize is False
