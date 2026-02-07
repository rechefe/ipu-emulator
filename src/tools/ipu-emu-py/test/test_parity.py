"""Phase 5 — C/Python emulator parity verification.

This module provides a 1:1 mapping of every C++ GTest in
``test/test_ipu_emulator.cpp`` to its Python equivalent, proving
behavioural parity between the C and Python emulators.

Each test is annotated with the original C++ test name.

C++ Test File                       Python Parity Test
============================        ========================================
RegisterOperations_SetLrRegister    TestCParity_Registers::test_set_lr
RegisterOperations_IncrementLr      TestCParity_Registers::test_incr_lr
RegisterOperations_DirectAccess     TestCParity_Registers::test_direct_access
RegisterOperations_AddLrLr          TestCParity_Registers::test_add_lr_lr
RegisterOperations_AddLrCr          TestCParity_Registers::test_add_lr_cr
RegisterOperations_SubLrLr          TestCParity_Registers::test_sub_lr_lr
RegisterOperations_SubCrLr          TestCParity_Registers::test_sub_cr_lr
RegisterOperations_AddSubOverflow   TestCParity_Registers::test_add_sub_overflow
Memory_LoadFromMemory               TestCParity_Memory::test_load_from_memory
Memory_StoreToMemory                TestCParity_Memory::test_store_to_memory
Memory_Fp8Conversions               TestCParity_Memory::test_fp8_conversions
Memory_CyclicRegisterLoad           TestCParity_Memory::test_cyclic_register_load
Memory_MaskRegisterLoad             TestCParity_Memory::test_mask_register_load
Memory_MaskAffectsMultiplication    TestCParity_Memory::test_mask_affects_mult
Memory_MaskWithShift                TestCParity_Memory::test_mask_with_shift
ControlFlow_UnconditionalBranch     TestCParity_ControlFlow::test_branch
ControlFlow_ConditionalBranchNE     TestCParity_ControlFlow::test_bne
SimpleLoop                          TestCParity_ControlFlow::test_loop
ProgramCounterTest                  TestCParity_Misc::test_pc
DirectRegisterAccess                TestCParity_Misc::test_direct_register_access
AccumulatorReset                    TestCParity_Misc::test_accumulator_reset

Additionally, the C++ fully_connected shell tests are matched by
``test_emulator_e2e.py::TestEndToEnd``:

fully_connected_test_int8           TestEndToEnd::test_fc_end_to_end[int8-INT8]
fully_connected_test_fp8_e4m3       TestEndToEnd::test_fc_end_to_end_fp8[fp8_e4m3-...]
fully_connected_test_fp8_e5m2       TestEndToEnd::test_fc_end_to_end_fp8[fp8_e5m2-...]

The Python test suite *exceeds* the C++ suite with additional coverage
of: beq, blt, bnz, bz branches; breakpoint variants (break, break.ifeq);
FP8 multiply accumulation; decode/encode round-trip; and run_test/debug.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from ipu_emu.execute import decode_instruction_word, execute_next_instruction, BreakResult
from ipu_emu.emulator import load_program, run_until_complete
from ipu_emu.ipu_state import IpuState
from ipu_emu.ipu_math import DType, fp32_to_fp8_bytes, fp8_bytes_to_fp32

from ipu_as.lark_tree import assemble


# ---------------------------------------------------------------------------
# Helpers (mirrors IpuTestHelper from ipu_test_helper.h)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_labels():
    from ipu_as.label import reset_labels
    reset_labels()
    yield
    reset_labels()


def _run(asm_code: str, max_cycles: int = 100_000) -> IpuState:
    """Assemble, load, execute — returns final state."""
    encoded = assemble(asm_code)
    decoded = [decode_instruction_word(w) for w in encoded]
    state = IpuState()
    load_program(state, decoded)
    run_until_complete(state, max_cycles=max_cycles)
    return state


def _make(asm_code: str) -> IpuState:
    """Assemble and load without running."""
    encoded = assemble(asm_code)
    decoded = [decode_instruction_word(w) for w in encoded]
    state = IpuState()
    load_program(state, decoded)
    return state


# ============================================================================
#  C++ RegisterOperations_* parity
# ============================================================================


class TestCParity_Registers:
    """1:1 port of the RegisterOperations_* C++ GTests."""

    def test_set_lr(self):
        """C++: RegisterOperations_SetLrRegister"""
        state = _run("set lr13 0x1000;;\nbkpt;;\n")
        assert state.regfile.get_lr(13) == 0x1000

    def test_incr_lr(self):
        """C++: RegisterOperations_IncrementLrRegister"""
        state = _run("set lr11 10;;\nincr lr11 5;;\nincr lr11 3;;\nbkpt;;\n")
        assert state.regfile.get_lr(11) == 18

    def test_direct_access(self):
        """C++: RegisterOperations_DirectAccess"""
        state = IpuState()
        state.regfile.set_lr(11, 0xDEADBEEF)
        state.regfile.set_lr(5, 0x12345678)
        assert state.regfile.get_lr(11) == 0xDEADBEEF
        assert state.regfile.get_lr(5) == 0x12345678
        state.regfile.set_cr(0, 0xABCDEF00)
        assert state.regfile.get_cr(0) == 0xABCDEF00

    def test_add_lr_lr(self):
        """C++: RegisterOperations_AddLrLr"""
        state = _run("set lr1 100;;\nset lr2 50;;\nadd lr3 lr1 lr2;;\nbkpt;;\n")
        assert state.regfile.get_lr(1) == 100
        assert state.regfile.get_lr(2) == 50
        assert state.regfile.get_lr(3) == 150

    def test_add_lr_cr(self):
        """C++: RegisterOperations_AddLrCr"""
        state = _make("set lr1 200;;\nadd lr4 lr1 cr5;;\nbkpt;;\n")
        state.regfile.set_cr(5, 75)
        run_until_complete(state)
        assert state.regfile.get_lr(1) == 200
        assert state.regfile.get_cr(5) == 75
        assert state.regfile.get_lr(4) == 275

    def test_sub_lr_lr(self):
        """C++: RegisterOperations_SubLrLr"""
        state = _run("set lr1 100;;\nset lr2 30;;\nsub lr3 lr1 lr2;;\nbkpt;;\n")
        assert state.regfile.get_lr(1) == 100
        assert state.regfile.get_lr(2) == 30
        assert state.regfile.get_lr(3) == 70

    def test_sub_cr_lr(self):
        """C++: RegisterOperations_SubCrLr"""
        state = _make("set lr2 45;;\nsub lr5 cr3 lr2;;\nbkpt;;\n")
        state.regfile.set_cr(3, 200)
        run_until_complete(state)
        assert state.regfile.get_cr(3) == 200
        assert state.regfile.get_lr(2) == 45
        assert state.regfile.get_lr(5) == 155

    def test_add_sub_overflow(self):
        """C++: RegisterOperations_AddSubOverflow"""
        state = _run(
            "set lr1 0xFFFF;;\n"
            "set lr2 1;;\n"
            "add lr3 lr1 lr2;;\n"
            "set lr4 0;;\n"
            "set lr5 1;;\n"
            "sub lr6 lr4 lr5;;\n"
            "bkpt;;\n"
        )
        assert state.regfile.get_lr(3) == 0x10000        # 0xFFFF + 1
        assert state.regfile.get_lr(6) == 0xFFFFFFFF      # 32-bit underflow


# ============================================================================
#  C++ Memory_* parity
# ============================================================================


class TestCParity_Memory:
    """1:1 port of the Memory_* C++ GTests."""

    def test_load_from_memory(self):
        """C++: Memory_LoadFromMemory"""
        test_data = bytes(range(128))
        state = _make("set lr13 0x1000;;\nldr_mult_reg r1 lr13 cr0;;\nbkpt;;\n")
        state.xmem.write_address(0x1000, test_data)
        run_until_complete(state)

        assert state.regfile.get_lr(13) == 0x1000
        assert state.regfile.get_r(1) == bytearray(test_data)

    def test_store_to_memory(self):
        """C++: Memory_StoreToMemory — mult.ee(2*3)→acc→store → 6 per word."""
        r1_data = bytes([2] * 128)
        cyclic_data = bytes([3] * 512)

        state = _make(
            "set lr13 0x1000;;\n"
            "ldr_mult_reg r1 lr13 cr0;;\n"
            "set lr14 0x2000;;\n"
            "set lr15 0;;\n"
            "ldr_cyclic_mult_reg lr14 cr0 lr15;;\n"
            "reset_acc;;\n"
            "mult.ee r1 lr0 lr0 lr0;\nacc;;\n"
            "set lr0 0x3000;;\n"
            "str_acc_reg lr0 cr0;;\n"
            "bkpt;;\n"
        )
        state.xmem.write_address(0x1000, r1_data)
        state.xmem.write_address(0x2000, cyclic_data)
        run_until_complete(state)

        acc_bytes = state.xmem.read_address(0x3000, 512)
        words = struct.unpack_from("<128I", acc_bytes)
        for i, w in enumerate(words):
            assert w == 6, f"acc word {i} should be 6, got {w}"

    def test_fp8_conversions(self):
        """C++: Memory_Fp8Conversions — FP32↔FP8_E4M3 round-trip."""
        values = [1.0, 2.0, 3.0, 4.0, 0.5, -1.0]
        raw = fp32_to_fp8_bytes(np.array(values, dtype=np.float32), DType.FP8_E4M3)
        back = fp8_bytes_to_fp32(raw, DType.FP8_E4M3)
        for i, (expected, actual) in enumerate(zip(values, back)):
            assert abs(actual - expected) < 0.1, f"Value {i}: {actual} vs {expected}"

    def test_cyclic_register_load(self):
        """C++: Memory_CyclicRegisterLoad"""
        cyclic_data = bytes([(i * 2) & 0xFF for i in range(128)])
        state = _make("set lr0 0x5000;;\nset lr1 0;;\nldr_cyclic_mult_reg lr0 cr0 lr1;;\nbkpt;;\n")
        state.xmem.write_address(0x5000, cyclic_data)
        run_until_complete(state)

        assert state.regfile.get_lr(0) == 0x5000
        assert state.regfile.get_r_cyclic_at(0, 128) == bytearray(cyclic_data)

    def test_mask_register_load(self):
        """C++: Memory_MaskRegisterLoad"""
        mask_data = bytes([(i + 1) & 0xFF for i in range(128)])
        state = _make("set lr0 0x6000;;\nldr_mult_mask_reg lr0 cr0;;\nbkpt;;\n")
        state.xmem.write_address(0x6000, mask_data)
        run_until_complete(state)

        assert state.regfile.get_lr(0) == 0x6000
        assert state.regfile.get_r_mask() == bytearray(mask_data)

    def test_mask_affects_mult(self):
        """C++: Memory_MaskAffectsMultiplication — first 64 bits masked out."""
        r0_data = bytes([2] * 128)
        cyclic_data = bytes([3] * 512)
        mask_data = bytearray(128)
        for i in range(8):
            mask_data[i] = 0xFF

        state = _make(
            "set lr0 0x1000;;\n"
            "ldr_mult_reg r0 lr0 cr0;;\n"
            "set lr1 0x2000;;\n"
            "set lr2 0;;\n"
            "ldr_cyclic_mult_reg lr1 cr0 lr2;;\n"
            "set lr3 0x3000;;\n"
            "ldr_mult_mask_reg lr3 cr0;;\n"
            "reset_acc;;\n"
            "set lr4 0;;\n"
            "set lr5 0;;\n"
            "set lr6 0;;\n"
            "mult.ee r0 lr6 lr4 lr5;\nacc;;\n"
            "set lr9 0x4000;;\n"
            "str_acc_reg lr9 cr0;;\n"
            "bkpt;;\n"
        )
        state.xmem.write_address(0x1000, r0_data)
        state.xmem.write_address(0x2000, cyclic_data)
        state.xmem.write_address(0x3000, mask_data)
        run_until_complete(state)

        acc_bytes = state.xmem.read_address(0x4000, 512)
        words = struct.unpack_from("<128I", acc_bytes)
        for i in range(64):
            assert words[i] == 0, f"word {i} should be masked to 0"
        for i in range(64, 128):
            assert words[i] == 6, f"word {i} should be 6"

    def test_mask_with_shift(self):
        """C++: Memory_MaskWithShift — mask index 1, shift 16 → bits 16-47 masked."""
        r0_data = bytes([5] * 128)
        cyclic_data = bytes([4] * 512)
        mask_data = bytearray(128)
        for i in range(16, 20):
            mask_data[i] = 0xFF

        state = _make(
            "set lr0 0x1000;;\n"
            "ldr_mult_reg r0 lr0 cr0;;\n"
            "set lr1 0x2000;;\n"
            "set lr2 0;;\n"
            "ldr_cyclic_mult_reg lr1 cr0 lr2;;\n"
            "set lr3 0x3000;;\n"
            "ldr_mult_mask_reg lr3 cr0;;\n"
            "reset_acc;;\n"
            "set lr4 1;;\n"
            "set lr5 16;;\n"
            "set lr6 0;;\n"
            "mult.ee r0 lr6 lr4 lr5;\nacc;;\n"
            "set lr9 0x4000;;\n"
            "str_acc_reg lr9 cr0;;\n"
            "bkpt;;\n"
        )
        state.xmem.write_address(0x1000, r0_data)
        state.xmem.write_address(0x2000, cyclic_data)
        state.xmem.write_address(0x3000, mask_data)
        run_until_complete(state)

        acc_bytes = state.xmem.read_address(0x4000, 512)
        words = struct.unpack_from("<128I", acc_bytes)
        for i in range(16):
            assert words[i] == 20, f"word {i} should be 20"
        for i in range(16, 48):
            assert words[i] == 0, f"word {i} should be masked to 0"
        for i in range(48, 128):
            assert words[i] == 20, f"word {i} should be 20"


# ============================================================================
#  C++ ControlFlow_* parity
# ============================================================================


class TestCParity_ControlFlow:
    """1:1 port of the ControlFlow_* / SimpleLoop C++ GTests."""

    def test_branch(self):
        """C++: ControlFlow_UnconditionalBranch + UnconditionalBranch (duplicate)"""
        state = _run(
            "set lr0 1;;\n"
            "b skip_section;;\n"
            "set lr0 2;;\n"
            "skip_section:\n"
            "set lr1 3;;\n"
            "bkpt;;\n"
        )
        assert state.regfile.get_lr(0) == 1
        assert state.regfile.get_lr(1) == 3

    def test_bne(self):
        """C++: ControlFlow_ConditionalBranchNotEqual"""
        state = _run(
            "set lr0 10;;\n"
            "set lr1 20;;\n"
            "bne lr0 lr1 not_equal_branch;;\n"
            "set lr2 0;;\n"
            "bkpt;;\n"
            "not_equal_branch:\n"
            "set lr2 1;;\n"
            "bkpt;;\n"
        )
        assert state.regfile.get_lr(2) == 1

    def test_loop(self):
        """C++: SimpleLoop — count to 10 via bne."""
        state = _run(
            "set lr0 0;;\n"
            "set lr1 10;;\n"
            "set lr2 0;;\n"
            "loop_start:\n"
            "incr lr0 1;;\n"
            "bne lr0 lr1 loop_start;;\n"
            "bkpt;;\n",
            max_cycles=1000,
        )
        assert state.regfile.get_lr(0) == 10


# ============================================================================
#  C++ ProgramCounterTest / DirectRegisterAccess / AccumulatorReset
# ============================================================================


class TestCParity_Misc:
    """Remaining C++ GTests."""

    def test_pc(self):
        """C++: ProgramCounterTest"""
        state = _make("set lr0 100;;\nset lr1 200;;\nbkpt;;\n")
        assert state.program_counter == 0
        execute_next_instruction(state)
        assert state.program_counter == 1
        execute_next_instruction(state)
        assert state.program_counter == 2

    def test_direct_register_access(self):
        """C++: DirectRegisterAccess"""
        state = IpuState()
        state.regfile.set_lr(0, 0xDEADBEEF)
        state.regfile.set_lr(5, 0x12345678)
        assert state.regfile.get_lr(0) == 0xDEADBEEF
        assert state.regfile.get_lr(5) == 0x12345678
        state.regfile.set_cr(0, 0xABCDEF00)
        assert state.regfile.get_cr(0) == 0xABCDEF00

    def test_accumulator_reset(self):
        """C++: AccumulatorReset"""
        state = _make("reset_acc;;\nbkpt;;\n")
        for i in range(128):
            state.regfile.set_r_acc_word(i, 12345)
        run_until_complete(state)
        for i in range(128):
            assert state.regfile.get_r_acc_word(i) == 0, f"acc word {i} should be zero"
