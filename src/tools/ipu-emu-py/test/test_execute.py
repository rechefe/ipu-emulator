"""Tests for the VLIW execution engine — Python port of test_ipu_emulator.cpp.

These tests use the assembler (``ipu_as.lark_tree.assemble``) to turn
inline assembly into encoded instruction words, load them into an
``IpuState``, execute, and check results.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from ipu_emu.execute import decode_instruction_word, BreakResult
from ipu_emu.emulator import (
    load_program,
    run_until_complete,
    run_with_debug,
    DebugAction,
)
from ipu_emu.ipu_state import IpuState, INST_MEM_SIZE
from ipu_emu.ipu_math import DType

from ipu_as.lark_tree import assemble


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(asm_code: str) -> IpuState:
    """Assemble *asm_code* and return a ready-to-run IpuState."""
    encoded = assemble(asm_code)
    decoded = [decode_instruction_word(w) for w in encoded]
    state = IpuState()
    load_program(state, decoded)
    return state


def _run(asm_code: str, **kw) -> IpuState:
    """Assemble, load, run, and return the final state."""
    state = _make_state(asm_code)
    run_until_complete(state, **kw)
    return state


# ============================================================================
# Basic Register Operations
# ============================================================================


class TestRegisterOperations:
    def test_set_lr(self):
        state = _run("set lr13 0x1000;;\nbkpt;;")
        assert state.regfile.get_lr(13) == 0x1000

    def test_incr_lr(self):
        state = _run(
            """\
set lr11 10;;
incr lr11 5;;
incr lr11 3;;
bkpt;;
"""
        )
        assert state.regfile.get_lr(11) == 18  # 10 + 5 + 3

    def test_direct_access(self):
        state = IpuState()
        state.regfile.set_lr(11, 0xDEADBEEF)
        state.regfile.set_lr(5, 0x12345678)
        assert state.regfile.get_lr(11) == 0xDEADBEEF
        assert state.regfile.get_lr(5) == 0x12345678
        state.regfile.set_cr(0, 0xABCDEF00)
        assert state.regfile.get_cr(0) == 0xABCDEF00

    def test_add_lr_lr(self):
        state = _run(
            """\
set lr1 100;;
set lr2 50;;
add lr3 lr1 lr2;;
bkpt;;
"""
        )
        assert state.regfile.get_lr(1) == 100
        assert state.regfile.get_lr(2) == 50
        assert state.regfile.get_lr(3) == 150

    def test_add_lr_cr(self):
        state = _make_state(
            """\
set lr1 200;;
add lr4 lr1 cr5;;
bkpt;;
"""
        )
        state.regfile.set_cr(5, 75)
        run_until_complete(state)
        assert state.regfile.get_lr(1) == 200
        assert state.regfile.get_cr(5) == 75
        assert state.regfile.get_lr(4) == 275

    def test_sub_lr_lr(self):
        state = _run(
            """\
set lr1 100;;
set lr2 30;;
sub lr3 lr1 lr2;;
bkpt;;
"""
        )
        assert state.regfile.get_lr(3) == 70

    def test_sub_cr_lr(self):
        state = _make_state(
            """\
set lr2 45;;
sub lr5 cr3 lr2;;
bkpt;;
"""
        )
        state.regfile.set_cr(3, 200)
        run_until_complete(state)
        assert state.regfile.get_lr(5) == 155


# ============================================================================
# Memory Operations
# ============================================================================


class TestMemoryOperations:
    def test_load_from_memory(self):
        test_data = bytes(range(128))
        state = _make_state(
            """\
set lr13 0x1000;;
ldr_mult_reg r1 lr13 cr0;;
bkpt;;
"""
        )
        state.xmem.write_address(0x1000, test_data)
        run_until_complete(state)

        assert state.regfile.get_lr(13) == 0x1000
        r1_data = state.regfile.get_r(1)
        assert r1_data == bytearray(test_data)

    def test_store_to_memory(self):
        """INT8: r1=all-2, cyclic=all-3, mult.ee → acc should be 6 per word."""
        r1_data = bytes([2] * 128)
        cyclic_data = bytes([3] * 512)

        state = _make_state(
            """\
set lr13 0x1000;;
ldr_mult_reg r1 lr13 cr0;;
set lr14 0x2000;;
set lr15 0;;
ldr_cyclic_mult_reg lr14 cr0 lr15;;
reset_acc;;
mult.ee r1 lr0 lr0 lr0;
acc;;
set lr0 0x3000;;
str_acc_reg lr0 cr0;;
bkpt;;
"""
        )
        state.xmem.write_address(0x1000, r1_data)
        state.xmem.write_address(0x2000, cyclic_data)
        run_until_complete(state)

        acc_bytes = state.xmem.read_address(0x3000, 512)
        words = struct.unpack_from("<128i", acc_bytes)
        for i, w in enumerate(words):
            assert w == 6, f"acc word {i} should be 6, got {w}"

    def test_cyclic_register_load(self):
        cyclic_data = bytes([(i * 2) & 0xFF for i in range(128)])
        state = _make_state(
            """\
set lr0 0x5000;;
set lr1 0;;
ldr_cyclic_mult_reg lr0 cr0 lr1;;
bkpt;;
"""
        )
        state.xmem.write_address(0x5000, cyclic_data)
        run_until_complete(state)

        assert state.regfile.get_lr(0) == 0x5000
        loaded = state.regfile.get_r_cyclic_at(0, 128)
        assert loaded == bytearray(cyclic_data)

    def test_mask_register_load(self):
        mask_data = bytes([(i + 1) & 0xFF for i in range(128)])
        state = _make_state(
            """\
set lr0 0x6000;;
ldr_mult_mask_reg lr0 cr0;;
bkpt;;
"""
        )
        state.xmem.write_address(0x6000, mask_data)
        run_until_complete(state)

        assert state.regfile.get_lr(0) == 0x6000
        loaded = state.regfile.get_r_mask()
        assert loaded == bytearray(mask_data)

    def test_mask_affects_multiplication(self):
        """First 64 bits of mask set → first 64 mult_res words zeroed."""
        r0_data = bytes([2] * 128)
        cyclic_data = bytes([3] * 512)
        # Mask: first 8 bytes = 0xFF (64 bits set), rest 0
        mask_data = bytearray(128)
        for i in range(8):
            mask_data[i] = 0xFF

        state = _make_state(
            """\
set lr0 0x1000;;
ldr_mult_reg r0 lr0 cr0;;
set lr1 0x2000;;
set lr2 0;;
ldr_cyclic_mult_reg lr1 cr0 lr2;;
set lr3 0x3000;;
ldr_mult_mask_reg lr3 cr0;;
reset_acc;;
set lr4 0;;
set lr5 0;;
set lr6 0;;
mult.ee r0 lr6 lr4 lr5;
acc;;
set lr9 0x4000;;
str_acc_reg lr9 cr0;;
bkpt;;
"""
        )
        state.xmem.write_address(0x1000, r0_data)
        state.xmem.write_address(0x2000, cyclic_data)
        state.xmem.write_address(0x3000, mask_data)
        run_until_complete(state)

        acc_bytes = state.xmem.read_address(0x4000, 512)
        words = struct.unpack_from("<128i", acc_bytes)
        for i in range(64):
            assert words[i] == 0, f"word {i} should be masked to 0"
        for i in range(64, 128):
            assert words[i] == 6, f"word {i} should be 6"

    def test_mask_with_shift(self):
        """Mask index 1, shift left 16 → bits 16-47 masked out."""
        r0_data = bytes([5] * 128)
        cyclic_data = bytes([4] * 512)
        # Mask index 1 is at bytes 16..31; set first 4 bytes = 32 bits
        mask_data = bytearray(128)
        for i in range(16, 20):
            mask_data[i] = 0xFF

        state = _make_state(
            """\
set lr0 0x1000;;
ldr_mult_reg r0 lr0 cr0;;
set lr1 0x2000;;
set lr2 0;;
ldr_cyclic_mult_reg lr1 cr0 lr2;;
set lr3 0x3000;;
ldr_mult_mask_reg lr3 cr0;;
reset_acc;;
set lr4 1;;
set lr5 16;;
set lr6 0;;
mult.ee r0 lr6 lr4 lr5;
acc;;
set lr9 0x4000;;
str_acc_reg lr9 cr0;;
bkpt;;
"""
        )
        state.xmem.write_address(0x1000, r0_data)
        state.xmem.write_address(0x2000, cyclic_data)
        state.xmem.write_address(0x3000, mask_data)
        run_until_complete(state)

        acc_bytes = state.xmem.read_address(0x4000, 512)
        words = struct.unpack_from("<128i", acc_bytes)
        for i in range(16):
            assert words[i] == 20, f"word {i} should be 20"
        for i in range(16, 48):
            assert words[i] == 0, f"word {i} should be masked to 0"
        for i in range(48, 128):
            assert words[i] == 20, f"word {i} should be 20"


# ============================================================================
# Control Flow
# ============================================================================


class TestControlFlow:
    def test_unconditional_branch(self):
        state = _run(
            """\
set lr0 1;;
b skip_section;;
set lr0 2;;
skip_section:
set lr1 3;;
bkpt;;
"""
        )
        assert state.regfile.get_lr(0) == 1
        assert state.regfile.get_lr(1) == 3

    def test_bne(self):
        state = _run(
            """\
set lr0 10;;
set lr1 20;;
bne lr0 lr1 not_equal_branch;;
set lr2 0;;
bkpt;;
not_equal_branch:
set lr2 1;;
bkpt;;
"""
        )
        assert state.regfile.get_lr(2) == 1

    def test_beq(self):
        state = _run(
            """\
set lr0 42;;
set lr1 42;;
beq lr0 lr1 equal_branch;;
set lr2 0;;
bkpt;;
equal_branch:
set lr2 1;;
bkpt;;
"""
        )
        assert state.regfile.get_lr(2) == 1

    def test_blt(self):
        state = _run(
            """\
set lr0 5;;
set lr1 6;;
blt lr0 lr1 less_branch;;
set lr2 0;;
bkpt;;
less_branch:
set lr2 1;;
bkpt;;
"""
        )
        assert state.regfile.get_lr(2) == 1

    def test_bnz(self):
        state = _run(
            """\
set lr0 1;;
bnz lr0 lr0 nonzero_branch;;
set lr2 0;;
bkpt;;
nonzero_branch:
set lr2 1;;
bkpt;;
"""
        )
        assert state.regfile.get_lr(2) == 1

    def test_bz(self):
        state = _run(
            """\
set lr0 0;;
bz lr0 lr0 zero_branch;;
set lr2 0;;
bkpt;;
zero_branch:
set lr2 1;;
bkpt;;
"""
        )
        assert state.regfile.get_lr(2) == 1

    def test_simple_loop(self):
        state = _run(
            """\
set lr0 0;;
set lr1 10;;
set lr2 0;;
loop_start:
incr lr0 1;;
bne lr0 lr1 loop_start;;
bkpt;;
""",
            max_cycles=1000,
        )
        assert state.regfile.get_lr(0) == 10

    def test_bkpt_halts(self):
        state = _run(
            """\
set lr0 99;;
bkpt;;
set lr0 0;;
"""
        )
        # bkpt sets PC = INST_MEM_SIZE, so `set lr0 0` is never reached
        assert state.regfile.get_lr(0) == 99


# ============================================================================
# Accumulator
# ============================================================================


class TestAccumulator:
    def test_reset(self):
        state = _make_state("reset_acc;;\nbkpt;;")
        # Pre-fill acc words with non-zero
        for i in range(128):
            state.regfile.set_r_acc_word(i, 12345)
        run_until_complete(state)
        for i in range(128):
            assert state.regfile.get_r_acc_word(i) == 0

    def test_acc_add_aaq(self):
        """acc.add_aaq adds the selected AAQ register (32-bit) to each of the 128 accumulator words."""
        state = _make_state(
            """\
reset_acc;;
acc.add_aaq aaq1;;
bkpt;;
"""
        )
        # INT8 dtype (cr15 = 0)
        state.regfile.set_cr(15, DType.INT8)
        # mult_res = 2 in each word; acc starts at 0
        for i in range(128):
            state.regfile.set_r_acc_word(i, 0)
        mult_buf = state.regfile.raw("mult_res")
        for i in range(128):
            struct.pack_into("<i", mult_buf, i * 4, 2)
        # aaq1 = 100 (added to every word)
        state.regfile.set_aaq(1, 100)
        run_until_complete(state)
        # Each word should be 0 + 2 + 100 = 102
        for i in range(128):
            w = state.regfile.get_r_acc_word(i)
            assert w == 102, f"word {i}: expected 102, got {w}"

    def test_acc_first(self):
        """acc.first sets r_acc to mult_res without adding previous r_acc."""
        state = _make_state(
            """\
acc.first;;
bkpt;;
"""
        )
        state.regfile.set_cr(15, DType.INT8)
        # Pre-fill acc with garbage; acc.first should ignore it
        for i in range(128):
            state.regfile.set_r_acc_word(i, 9999)
        mult_buf = state.regfile.raw("mult_res")
        for i in range(128):
            struct.pack_into("<i", mult_buf, i * 4, 7)
        run_until_complete(state)
        for i in range(128):
            assert state.regfile.get_r_acc_word(i) == 7, f"word {i}: expected 7, got {state.regfile.get_r_acc_word(i)}"

    def test_acc_add_aaq_first(self):
        """acc.add_aaq.first sets r_acc to mult_res + aaq (no previous sum)."""
        state = _make_state(
            """\
acc.add_aaq.first aaq2;;
bkpt;;
"""
        )
        state.regfile.set_cr(15, DType.INT8)
        for i in range(128):
            state.regfile.set_r_acc_word(i, 9999)  # should be ignored
        mult_buf = state.regfile.raw("mult_res")
        for i in range(128):
            struct.pack_into("<i", mult_buf, i * 4, 3)
        state.regfile.set_aaq(2, 10)
        run_until_complete(state)
        for i in range(128):
            w = state.regfile.get_r_acc_word(i)
            assert w == 13, f"word {i}: expected 13 (3+10), got {w}"

    def test_acc_max(self):
        """acc.max sets r_acc[i] = max(r_acc[i], mult_res[i], aaq_reg)."""
        state = _make_state(
            """\
acc.max aaq1;;
bkpt;;
"""
        )
        state.regfile.set_cr(15, DType.INT8)
        # r_acc: 1, mult_res: 2, aaq1: 3 → max(1,2,3)=3
        for i in range(128):
            state.regfile.set_r_acc_word(i, 1)
        mult_buf = state.regfile.raw("mult_res")
        for i in range(128):
            struct.pack_into("<i", mult_buf, i * 4, 2)
        state.regfile.set_aaq(1, 3)
        run_until_complete(state)
        for i in range(128):
            w = state.regfile.get_r_acc_word(i)
            assert w == 3, f"word {i}: expected max(1,2,3)=3, got {w}"

    def test_acc_max_signed(self):
        """acc.max treats register values as signed int32; negative values compare correctly."""
        state = _make_state(
            """\
acc.max aaq0;;
bkpt;;
"""
        )
        state.regfile.set_cr(15, DType.INT8)
        # r_acc: -10, mult_res: 2, aaq0: 5 → max(-10, 2, 5) = 5 (signed comparison)
        acc_buf = state.regfile.raw("r_acc")
        for i in range(128):
            struct.pack_into("<i", acc_buf, i * 4, -10)
        mult_buf = state.regfile.raw("mult_res")
        for i in range(128):
            struct.pack_into("<i", mult_buf, i * 4, 2)
        state.regfile.set_aaq(0, struct.unpack("<I", struct.pack("<i", 5))[0])
        run_until_complete(state)
        for i in range(128):
            val = struct.unpack_from("<i", state.regfile.raw("r_acc"), i * 4)[0]
            assert val == 5, f"word {i}: expected max(-10, 2, 5)=5 (signed), got {val}"

    def test_acc_max_first(self):
        """acc.max.first sets r_acc[i] = max(mult_res[i], aaq_reg); previous r_acc ignored."""
        state = _make_state(
            """\
acc.max.first aaq0;;
bkpt;;
"""
        )
        state.regfile.set_cr(15, DType.INT8)
        for i in range(128):
            state.regfile.set_r_acc_word(i, 9999)  # should be ignored
        mult_buf = state.regfile.raw("mult_res")
        for i in range(128):
            struct.pack_into("<i", mult_buf, i * 4, 5)
        state.regfile.set_aaq(0, 10)
        run_until_complete(state)
        for i in range(128):
            w = state.regfile.get_r_acc_word(i)
            assert w == 10, f"word {i}: expected max(5,10)=10, got {w}"

    def test_acc_stride_no_stride(self):
        """acc.stride with both strides off copies all 128 mult_res words to r_acc from start 0."""
        state = _make_state(
            """\
set lr0 0;;
acc.stride 8 off off lr0;;
bkpt;;
"""
        )
        state.regfile.set_cr(15, DType.INT8)
        mult_buf = state.regfile.raw("mult_res")
        for i in range(128):
            struct.pack_into("<i", mult_buf, i * 4, i)
        run_until_complete(state)
        for i in range(128):
            w = state.regfile.get_r_acc_word(i)
            assert w == i, f"word {i}: expected {i}, got {w}"

    def test_acc_stride_horizontal_no_expand(self):
        """acc.stride with horizontal on, no expand: take every 2nd column → 64 elements at r_acc[0:64]."""
        state = _make_state(
            """\
set lr0 0;;
acc.stride 8 on off lr0;;
bkpt;;
"""
        )
        state.regfile.set_cr(15, DType.INT8)
        mult_buf = state.regfile.raw("mult_res")
        for i in range(128):
            struct.pack_into("<i", mult_buf, i * 4, i)
        run_until_complete(state)
        # Rows of 8: even columns 0,2,4,6 → indices 0,2,4,6, 8,10,12,14, ...
        for out_i in range(64):
            row = out_i // 4
            col = (out_i % 4) * 2
            expected = row * 8 + col
            w = state.regfile.get_r_acc_word(out_i)
            assert w == expected, f"out[{out_i}]: expected {expected}, got {w}"

    def test_acc_stride_offset(self):
        """acc.stride with offset: (lr0 % 4)*32 is start index; 64 elements written at r_acc[32:96]."""
        state = _make_state(
            """\
set lr0 1;;
acc.stride 8 on off lr0;;
bkpt;;
"""
        )
        # lr0=1 → offset % 4 = 1 → start index 32. Horizontal on, no expand → 64 elements.
        state.regfile.set_cr(15, DType.INT8)
        for i in range(128):
            state.regfile.set_r_acc_word(i, 0)
        mult_buf = state.regfile.raw("mult_res")
        for i in range(128):
            struct.pack_into("<i", mult_buf, i * 4, 100 + i)
        run_until_complete(state)
        for i in range(32):
            w = state.regfile.get_r_acc_word(i)
            assert w == 0, f"word {i} (before start): expected 0, got {w}"
        for out_i in range(64):
            row = out_i // 4
            col = (out_i % 4) * 2
            expected_src = row * 8 + col
            w = state.regfile.get_r_acc_word(32 + out_i)
            assert w == 100 + expected_src, f"word {32 + out_i}: expected {100 + expected_src}, got {w}"
        for i in range(96, 128):
            w = state.regfile.get_r_acc_word(i)
            assert w == 0, f"word {i} (after segment): expected 0, got {w}"

    def test_acc_agg_sum_value(self):
        """agg sum value: sum of 128 r_acc words, identity post fn, store to aaq0."""
        import struct

        state = _make_state(
            """\
agg sum value cr0 aaq0;;
bkpt;;
"""
        )
        state.regfile.set_cr(15, DType.INT8)
        # r_acc: set each word to 1, so sum = 128
        for i in range(128):
            state.regfile.set_r_acc_word(i, struct.unpack("<I", struct.pack("<i", 1))[0])
        state.regfile.set_aaq(0, 0)

        run_until_complete(state)
        # Sum of 128 ones = 128
        assert state.regfile.get_aaq(0) == struct.unpack("<I", struct.pack("<i", 128))[0]

    def test_acc_agg_max_value(self):
        """agg max value: max of 128 r_acc words and current aaq, store to aaq1."""
        import struct

        state = _make_state(
            """\
agg max value cr0 aaq1;;
bkpt;;
"""
        )
        state.regfile.set_cr(15, DType.INT8)
        for i in range(128):
            state.regfile.set_r_acc_word(i, struct.unpack("<I", struct.pack("<i", 10 + (i % 5)))[0])
        state.regfile.set_aaq(1, struct.unpack("<I", struct.pack("<i", 20))[0])  # 20 > 14

        run_until_complete(state)
        # Max of r_acc is 14, but aaq1 was 20, so max(..., 20) = 20
        assert state.regfile.get_aaq(1) == struct.unpack("<I", struct.pack("<i", 20))[0]

    def test_acc_agg_max_value_updates_when_larger(self):
        """agg max value: when r_acc has a value larger than aaq, aaq is updated."""
        import struct

        state = _make_state(
            """\
agg max value cr0 aaq0;;
bkpt;;
"""
        )
        state.regfile.set_cr(15, DType.INT8)
        for i in range(128):
            state.regfile.set_r_acc_word(i, struct.unpack("<I", struct.pack("<i", 5))[0])
        state.regfile.set_r_acc_word(10, struct.unpack("<I", struct.pack("<i", 100))[0])
        state.regfile.set_aaq(0, struct.unpack("<I", struct.pack("<i", 0))[0])

        run_until_complete(state)
        assert state.regfile.get_aaq(0) == struct.unpack("<I", struct.pack("<i", 100))[0]

    def test_acc_agg_sum_value_cr(self):
        """agg sum value_cr: result = sum(r_acc) * cr[cr_idx]."""
        import struct

        state = _make_state(
            """\
agg sum value_cr cr1 aaq2;;
bkpt;;
"""
        )
        state.regfile.set_cr(15, DType.INT8)
        for i in range(128):
            state.regfile.set_r_acc_word(i, struct.unpack("<I", struct.pack("<i", 1))[0])
        state.regfile.set_cr(1, struct.unpack("<I", struct.pack("<i", 3))[0])
        state.regfile.set_aaq(2, 0)

        run_until_complete(state)
        # sum = 128, 128 * 3 = 384
        assert state.regfile.get_aaq(2) == struct.unpack("<I", struct.pack("<i", 384))[0]


# ============================================================================


class TestProgramCounter:
    def test_pc_advances(self):
        from ipu_emu.execute import execute_next_instruction

        state = _make_state(
            """\
set lr0 100;;
set lr1 200;;
bkpt;;
"""
        )
        assert state.program_counter == 0
        execute_next_instruction(state)
        assert state.program_counter == 1
        execute_next_instruction(state)
        assert state.program_counter == 2


# ============================================================================
# Breakpoints
# ============================================================================


class TestBreakpoints:
    def test_break_stops_execution(self):
        from ipu_emu.execute import execute_next_instruction

        state = _make_state(
            """\
set lr0 1;;
break;;
set lr0 2;;
bkpt;;
"""
        )
        r = execute_next_instruction(state)
        assert r == BreakResult.CONTINUE
        assert state.regfile.get_lr(0) == 1

        r = execute_next_instruction(state)
        assert r == BreakResult.BREAK

    def test_break_ifeq(self):
        from ipu_emu.execute import execute_next_instruction

        state = _make_state(
            """\
set lr5 42;;
break.ifeq lr5 42;;
bkpt;;
"""
        )
        execute_next_instruction(state)  # set lr5 42
        assert state.regfile.get_lr(5) == 42

        r = execute_next_instruction(state)
        assert r == BreakResult.BREAK

    def test_break_ifeq_no_match(self):
        from ipu_emu.execute import execute_next_instruction

        state = _make_state(
            """\
set lr5 10;;
break.ifeq lr5 42;;
bkpt;;
"""
        )
        execute_next_instruction(state)
        r = execute_next_instruction(state)
        assert r == BreakResult.CONTINUE

    def test_run_with_debug_step(self):
        actions = iter([DebugAction.STEP, DebugAction.CONTINUE])

        def callback(state, cycle):
            return next(actions, DebugAction.CONTINUE)

        state = _make_state(
            """\
break;;
set lr0 1;;
bkpt;;
"""
        )
        run_with_debug(state, callback)
        assert state.regfile.get_lr(0) == 1


# ============================================================================
# Decode / encode round-trip
# ============================================================================


class TestDecodeRoundtrip:
    def test_nop_decodes(self):
        """An all-NOP instruction should decode without error."""
        fields = decode_instruction_word(0)
        assert isinstance(fields, dict)
        assert len(fields) > 0

    def test_roundtrip(self):
        """Assemble → decode → verify known fields."""
        encoded = assemble("set lr13 0x1000;;\nbkpt;;")
        assert len(encoded) == 2
        d = decode_instruction_word(encoded[0])
        # LR opcode should be 'set' = index 1
        assert d["lr_inst_0_token_0_lr_inst_opcode"] == 1  # set
        assert d["lr_inst_0_token_1_lr_reg_field"] == 13
        assert d["lr_inst_0_token_4_lr_immediate_type"] == 0x1000


# ============================================================================
# FP8 multiply
# ============================================================================


class TestFp8:
    def test_fp8_e4m3_mult_ee(self):
        """FP8 E4M3: 1.0 × 2.0 → 2.0 for every element."""
        from ml_dtypes import float8_e4m3fn

        fp_one_byte = int(
            np.array(1.0, dtype=np.float32).astype(float8_e4m3fn).view(np.uint8).item()
        )
        fp_two_byte = int(
            np.array(2.0, dtype=np.float32).astype(float8_e4m3fn).view(np.uint8).item()
        )

        r0_data = bytes([fp_one_byte] * 128)
        cyclic_data = bytes([fp_two_byte] * 512)

        state = _make_state(
            """\
set lr0 0x1000;;
ldr_mult_reg r0 lr0 cr0;;
set lr1 0x2000;;
set lr2 0;;
ldr_cyclic_mult_reg lr1 cr0 lr2;;
reset_acc;;
set lr4 0;;
set lr5 0;;
set lr6 0;;
mult.ee r0 lr6 lr4 lr5;
acc;;
bkpt;;
"""
        )
        # Set dtype to FP8_E4M3
        state.set_cr_dtype(DType.FP8_E4M3)
        state.xmem.write_address(0x1000, r0_data)
        state.xmem.write_address(0x2000, cyclic_data)
        run_until_complete(state)

        # Read back acc as floats
        acc_raw = state.regfile.raw("r_acc")
        for i in range(128):
            val = struct.unpack_from("<f", acc_raw, i * 4)[0]
            assert abs(val - 2.0) < 0.01, f"acc word {i}: expected 2.0, got {val}"


# ============================================================================
# mult.ve.cr and mult.ve.aaq
# ============================================================================


class TestMultVeCrAaq:
    """Tests for mult.ve.cr and mult.ve.aaq instructions."""

    # ------------------------------------------------------------------
    # mult.ve.cr
    # ------------------------------------------------------------------

    def test_mult_ve_cr_int8(self):
        """mult.ve.cr INT8: scalar from CR × RC elements."""
        # CR scalar byte = 3, RC elements = all 2 → each result = 3*2 = 6
        cyclic_data = bytes([2] * 512)

        state = _make_state(
            """\
set lr0 0x1000;;
set lr1 0;;
ldr_cyclic_mult_reg lr0 cr0 lr1;;
reset_acc;;
set lr2 0;;
set lr3 0;;
set lr4 0;;
mult.ve.cr lr2 lr3 lr4 cr1;
acc;;
bkpt;;
"""
        )
        state.regfile.set_cr(15, DType.INT8)
        state.regfile.set_cr(1, 3)  # scalar = low byte = 3
        state.xmem.write_address(0x1000, cyclic_data)
        run_until_complete(state)

        acc_raw = state.regfile.raw("r_acc")
        for i in range(128):
            val = struct.unpack_from("<i", acc_raw, i * 4)[0]
            assert val == 6, f"acc word {i}: expected 6, got {val}"

    def test_mult_ve_cr_negative_int8(self):
        """mult.ve.cr INT8: signed negative scalar × positive RC elements."""
        # CR scalar byte = 0xFE = -2 (signed int8), RC elements = 5 → result = -10
        cyclic_data = bytes([5] * 512)

        state = _make_state(
            """\
set lr0 0x1000;;
set lr1 0;;
ldr_cyclic_mult_reg lr0 cr0 lr1;;
reset_acc;;
set lr2 0;;
set lr3 0;;
set lr4 0;;
mult.ve.cr lr2 lr3 lr4 cr2;
acc;;
bkpt;;
"""
        )
        state.regfile.set_cr(15, DType.INT8)
        state.regfile.set_cr(2, 0xFE)  # low byte 0xFE = int8(-2)
        state.xmem.write_address(0x1000, cyclic_data)
        run_until_complete(state)

        acc_raw = state.regfile.raw("r_acc")
        for i in range(128):
            val = struct.unpack_from("<i", acc_raw, i * 4)[0]
            assert val == -10, f"acc word {i}: expected -10, got {val}"

    def test_mult_ve_cr_boundary_padding(self):
        """mult.ve.cr: elements beyond RC boundary (512 bytes) are padded with int8 1."""
        # cyclic_offset = 450, so first 62 bytes come from RC, remaining 66 are padded with 1
        rc_fill = 4  # RC filled with 4
        scalar = 7
        pad_start = 62  # 512 - 450 = 62 elements in bounds

        state = _make_state(
            """\
reset_acc;;
set lr2 450;;
set lr3 0;;
set lr4 0;;
mult.ve.cr lr2 lr3 lr4 cr3;
acc;;
bkpt;;
"""
        )
        state.regfile.set_cr(15, DType.INT8)
        state.regfile.set_cr(3, scalar)
        # Fill the full 512-byte cyclic register directly
        state.regfile.set_r_cyclic_at(0, bytes([rc_fill] * 512))
        run_until_complete(state)

        acc_raw = state.regfile.raw("r_acc")
        for i in range(pad_start):
            val = struct.unpack_from("<i", acc_raw, i * 4)[0]
            assert val == scalar * rc_fill, f"word {i}: expected {scalar * rc_fill}, got {val}"
        for i in range(pad_start, 128):
            val = struct.unpack_from("<i", acc_raw, i * 4)[0]
            assert val == scalar * 1, f"word {i} (padded): expected {scalar}, got {val}"

    def test_mult_ve_cr_fp8e4m3(self):
        """mult.ve.cr FP8_E4M3: scalar 1.0 × RC elements 1.0 → result 1.0 each."""
        import numpy as np
        from ml_dtypes import float8_e4m3fn

        one_fp8 = int(np.array(1.0, dtype=np.float32).astype(float8_e4m3fn).view(np.uint8))
        cyclic_data = bytes([one_fp8] * 512)

        state = _make_state(
            """\
set lr0 0x1000;;
set lr1 0;;
ldr_cyclic_mult_reg lr0 cr0 lr1;;
reset_acc;;
set lr2 0;;
set lr3 0;;
set lr4 0;;
mult.ve.cr lr2 lr3 lr4 cr5;
acc;;
bkpt;;
"""
        )
        state.regfile.set_cr(15, DType.FP8_E4M3)
        state.regfile.set_cr(5, one_fp8)  # scalar = 1.0 in FP8_E4M3
        state.xmem.write_address(0x1000, cyclic_data)
        run_until_complete(state)

        acc_raw = state.regfile.raw("r_acc")
        for i in range(128):
            val = struct.unpack_from("<f", acc_raw, i * 4)[0]
            assert abs(val - 1.0) < 0.01, f"acc word {i}: expected 1.0, got {val}"

    def test_mult_ve_cr_boundary_padding_fp8e5m2(self):
        """mult.ve.cr FP8_E5M2: boundary elements padded with FP8 1.0."""
        import numpy as np
        from ml_dtypes import float8_e5m2

        two_fp8 = int(np.array(2.0, dtype=np.float32).astype(float8_e5m2).view(np.uint8))
        scalar_fp8 = int(np.array(3.0, dtype=np.float32).astype(float8_e5m2).view(np.uint8))

        state = _make_state(
            """\
reset_acc;;
set lr2 500;;
set lr3 0;;
set lr4 0;;
mult.ve.cr lr2 lr3 lr4 cr6;
acc;;
bkpt;;
"""
        )
        state.regfile.set_cr(15, DType.FP8_E5M2)
        state.regfile.set_cr(6, scalar_fp8)  # scalar = 3.0
        # Fill the full 512-byte cyclic register directly with 2.0 in FP8_E5M2
        state.regfile.set_r_cyclic_at(0, bytes([two_fp8] * 512))
        run_until_complete(state)

        acc_raw = state.regfile.raw("r_acc")
        for i in range(12):  # in-bounds (500..511): 3.0 * 2.0 = 6.0
            val = struct.unpack_from("<f", acc_raw, i * 4)[0]
            assert abs(val - 6.0) < 0.1, f"word {i}: expected 6.0, got {val}"
        for i in range(12, 128):  # padded (>=512): 3.0 * 1.0 = 3.0
            val = struct.unpack_from("<f", acc_raw, i * 4)[0]
            assert abs(val - 3.0) < 0.1, f"word {i} (padded): expected 3.0, got {val}"

    # ------------------------------------------------------------------
    # mult.ve.aaq
    # ------------------------------------------------------------------

    def test_mult_ve_aaq_int8(self):
        """mult.ve.aaq INT8: scalar from AAQ × RC elements."""
        # AAQ scalar byte = 5, RC elements = all 3 → each result = 15
        cyclic_data = bytes([3] * 512)

        state = _make_state(
            """\
set lr0 0x1000;;
set lr1 0;;
ldr_cyclic_mult_reg lr0 cr0 lr1;;
reset_acc;;
set lr2 0;;
set lr3 0;;
set lr4 0;;
mult.ve.aaq lr2 lr3 lr4 aaq0;
acc;;
bkpt;;
"""
        )
        state.regfile.set_cr(15, DType.INT8)
        state.regfile.set_aaq(0, 5)  # low byte = 5
        state.xmem.write_address(0x1000, cyclic_data)
        run_until_complete(state)

        acc_raw = state.regfile.raw("r_acc")
        for i in range(128):
            val = struct.unpack_from("<i", acc_raw, i * 4)[0]
            assert val == 15, f"acc word {i}: expected 15, got {val}"

    def test_mult_ve_aaq_boundary_padding(self):
        """mult.ve.aaq: elements beyond RC boundary are padded with int8 1."""
        scalar = 2
        in_bounds = 112  # offset=400, 512-400=112 in bounds, 16 padded

        state = _make_state(
            """\
reset_acc;;
set lr2 400;;
set lr3 0;;
set lr4 0;;
mult.ve.aaq lr2 lr3 lr4 aaq1;
acc;;
bkpt;;
"""
        )
        state.regfile.set_cr(15, DType.INT8)
        state.regfile.set_aaq(1, scalar)
        # Fill the full 512-byte cyclic register directly
        state.regfile.set_r_cyclic_at(0, bytes([10] * 512))
        run_until_complete(state)

        acc_raw = state.regfile.raw("r_acc")
        for i in range(in_bounds):
            val = struct.unpack_from("<i", acc_raw, i * 4)[0]
            assert val == scalar * 10, f"word {i}: expected {scalar * 10}, got {val}"
        for i in range(in_bounds, 128):
            val = struct.unpack_from("<i", acc_raw, i * 4)[0]
            assert val == scalar * 1, f"word {i} (padded): expected {scalar}, got {val}"

    def test_mult_ve_aaq_no_boundary(self):
        """mult.ve.aaq: when cyclic_offset+128 <= 512, no padding applied."""
        cyclic_data = bytes([7] * 512)
        scalar = 3

        state = _make_state(
            """\
set lr0 0x1000;;
set lr1 0;;
ldr_cyclic_mult_reg lr0 cr0 lr1;;
reset_acc;;
set lr2 0;;
set lr3 0;;
set lr4 0;;
mult.ve.aaq lr2 lr3 lr4 aaq2;
acc;;
bkpt;;
"""
        )
        state.regfile.set_cr(15, DType.INT8)
        state.regfile.set_aaq(2, scalar)
        state.xmem.write_address(0x1000, cyclic_data)
        run_until_complete(state)

        acc_raw = state.regfile.raw("r_acc")
        for i in range(128):
            val = struct.unpack_from("<i", acc_raw, i * 4)[0]
            assert val == scalar * 7, f"acc word {i}: expected {scalar * 7}, got {val}"

    # ------------------------------------------------------------------
    # Backward compatibility: existing instructions unaffected
    # ------------------------------------------------------------------

    def test_backward_compat_mult_ee(self):
        """mult.ee still works correctly after adding new mult variants."""
        r0_data = bytes([4] * 128)
        cyclic_data = bytes([5] * 512)

        state = _make_state(
            """\
set lr0 0x1000;;
ldr_mult_reg r0 lr0 cr0;;
set lr1 0x2000;;
set lr2 0;;
ldr_cyclic_mult_reg lr1 cr0 lr2;;
reset_acc;;
mult.ee r0 lr2 lr2 lr2;
acc;;
bkpt;;
"""
        )
        state.regfile.set_cr(15, DType.INT8)
        state.xmem.write_address(0x1000, r0_data)
        state.xmem.write_address(0x2000, cyclic_data)
        run_until_complete(state)

        acc_raw = state.regfile.raw("r_acc")
        for i in range(128):
            val = struct.unpack_from("<i", acc_raw, i * 4)[0]
            assert val == 20, f"acc word {i}: expected 20, got {val}"

    def test_backward_compat_mult_ve(self):
        """mult.ve still works correctly after adding new mult variants."""
        cyclic_data = bytes([6] * 512)
        r0_data = bytes([0] * 128)
        r0_data = bytearray(r0_data)
        r0_data[0] = 3  # fixed_ra_idx=0 → scalar = 3

        state = _make_state(
            """\
set lr0 0x1000;;
ldr_mult_reg r0 lr0 cr0;;
set lr1 0x2000;;
set lr2 0;;
ldr_cyclic_mult_reg lr1 cr0 lr2;;
reset_acc;;
mult.ve r0 lr2 lr2 lr2 lr2;
acc;;
bkpt;;
"""
        )
        state.regfile.set_cr(15, DType.INT8)
        state.xmem.write_address(0x1000, bytes(r0_data))
        state.xmem.write_address(0x2000, cyclic_data)
        run_until_complete(state)

        acc_raw = state.regfile.raw("r_acc")
        for i in range(128):
            val = struct.unpack_from("<i", acc_raw, i * 4)[0]
            assert val == 18, f"acc word {i}: expected 18, got {val}"
