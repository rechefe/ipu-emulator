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

from ipu_as.lark_tree import assemble, parse

from ipu_as.compound_inst import CompoundInst

from ipu_common.activations import ACTIVATION_FN_NAMES, apply_activation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(
    asm_code: str,
    *,
    cr: dict[int, int] | None = None,
    leaky_relu_alpha: float | None = None,
    elu_alpha: float | None = None,
    prelu_alpha: float | None = None,
) -> IpuState:
    """Assemble *asm_code* and return a ready-to-run IpuState.

    Optional *cr* presets configuration registers before the program is loaded
    (e.g. constants read by ``SET lr* cr*``). Optional α kwargs are forwarded to
    :class:`IpuState` (emulator-only; not CR).
    """
    encoded = assemble(asm_code)
    decoded = [decode_instruction_word(w) for w in encoded]
    state = IpuState(
        leaky_relu_alpha=leaky_relu_alpha,
        elu_alpha=elu_alpha,
        prelu_alpha=prelu_alpha,
    )
    if cr:
        for idx, val in cr.items():
            state.regfile.set_cr(idx, val)
    load_program(state, decoded)
    return state


def _run(
    asm_code: str,
    *,
    cr: dict[int, int] | None = None,
    leaky_relu_alpha: float | None = None,
    elu_alpha: float | None = None,
    prelu_alpha: float | None = None,
    max_cycles: int = 100_000,
) -> IpuState:
    """Assemble, load, run, and return the final state."""
    state = _make_state(
        asm_code,
        cr=cr,
        leaky_relu_alpha=leaky_relu_alpha,
        elu_alpha=elu_alpha,
        prelu_alpha=prelu_alpha,
    )
    run_until_complete(state, max_cycles)
    return state


# ============================================================================
# Basic Register Operations
# ============================================================================


class TestRegisterOperations:
    def test_set_lr(self):
        state = _run("SET lr13 cr8;;\nBKPT;;", cr={8: 0x1000})
        assert state.regfile.get_lr(13) == 0x1000

    def test_add_imm_accumulates_lr(self):
        state = _run("""\
SET lr11 cr8;;
ADD lr11 lr11 5;;
ADD lr11 lr11 3;;
BKPT;;
""",
            cr={8: 10})
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
        state = _run("""\
SET lr1 cr8;;
SET lr2 cr9;;
ADD lr3 lr1 lr2;;
BKPT;;
""",
            cr={8: 100, 9: 50})
        assert state.regfile.get_lr(1) == 100
        assert state.regfile.get_lr(2) == 50
        assert state.regfile.get_lr(3) == 150

    def test_add_lr_cr(self):
        state = _make_state("""\
SET lr1 cr8;;
ADD lr4 lr1 cr5;;
BKPT;;
""",
            cr={8: 200})
        state.regfile.set_cr(5, 75)
        run_until_complete(state)
        assert state.regfile.get_lr(1) == 200
        assert state.regfile.get_cr(5) == 75
        assert state.regfile.get_lr(4) == 275

    def test_add_lr_lr_imm5(self):
        state = _run("""\
SET lr1 cr8;;
ADD lr4 lr1 11;;
BKPT;;
""",
            cr={8: 200})
        assert state.regfile.get_lr(4) == 211

    def test_sub_lr_lr(self):
        state = _run("""\
SET lr1 cr8;;
SET lr2 cr9;;
SUB lr3 lr1 lr2;;
BKPT;;
""",
            cr={8: 100, 9: 30})
        assert state.regfile.get_lr(3) == 70

    def test_sub_lr_lr_cr(self):
        state = _make_state("""\
SET lr2 cr8;;
SUB lr5 lr2 cr3;;
BKPT;;
""",
            cr={8: 45})
        state.regfile.set_cr(3, 200)
        run_until_complete(state)
        assert state.regfile.get_lr(5) == (45 - 200) & 0xFFFFFFFF

    def test_sub_lr_lr_imm5(self):
        state = _run("""\
SET lr2 cr8;;
SUB lr3 lr2 30;;
BKPT;;
""",
            cr={8: 100})
        assert state.regfile.get_lr(3) == 70


# ============================================================================
# INCR_MOD_POW2 (Issue #47): dst <- (dst + step) mod 2^k
# ============================================================================


class TestIncrModPow2:
    def test_basic_wrap_small_k(self):
        """k=4 → mod 16; 15 + 1 wraps to 0."""
        state = _run("""\
SET lr0 cr8;;
SET lr1 cr9;;
INCR_MOD_POW2 lr0 lr1 4;;
BKPT;;
""",
            cr={8: 15, 9: 1})
        assert state.regfile.get_lr(0) == 0

    def test_k9_mask_511(self):
        """k=9 → mask 511; largest legal k from spec."""
        state = _run("""\
SET lr0 cr8;;
SET lr1 cr9;;
INCR_MOD_POW2 lr0 lr1 9;;
BKPT;;
""",
            cr={8: 500, 9: 20})
        assert state.regfile.get_lr(0) == (500 + 20) & 511

    def test_read_before_write_same_register(self):
        """dst and step both lr0: uses snapshot value of lr0 for the sum."""
        state = _run("""\
SET lr0 cr8;;
INCR_MOD_POW2 lr0 lr0 3;;
BKPT;;
""",
            cr={8: 5})
        assert state.regfile.get_lr(0) == (5 + 5) & 7

    def test_step_from_cr(self):
        state = _make_state("""\
SET lr0 cr8;;
INCR_MOD_POW2 lr0 cr4 4;;
BKPT;;
""",
            cr={8: 2})
        state.regfile.set_cr(4, 10)
        run_until_complete(state)
        assert state.regfile.get_lr(0) == (2 + 10) & 15

    def test_large_unsigned_step_reduces_mod_mask(self):
        """Uint32 add before mask: step loaded from CR as full uint32."""
        state = _make_state("""\
SET lr0 cr8;;
INCR_MOD_POW2 lr0 cr5 3;;
BKPT;;
""",
            cr={8: 3})
        state.regfile.set_cr(5, 0xFFFFFFFE)
        run_until_complete(state)
        assert state.regfile.get_lr(0) == (3 + 0xFFFFFFFE) & 7

    def test_k_encoded_four_bits(self):
        """k operand uses 4 bits: semantic k=9 encodes as 8 in the instruction word."""
        encoded = assemble("INCR_MOD_POW2 lr0 lr1 9;; BKPT;;")
        d = decode_instruction_word(encoded[0])
        assert d["lr_inst_0_token_3_lr_reg_field"] == 8

    def test_assembler_rejects_k_out_of_range(self):
        with pytest.raises(ValueError, match=r"INCR_MOD_POW2 k operand"):
            CompoundInst(parse("INCR_MOD_POW2 lr0 lr1 0;;")[0])
        with pytest.raises(ValueError, match=r"INCR_MOD_POW2 k operand"):
            CompoundInst(parse("INCR_MOD_POW2 lr0 lr1 10;;")[0])

    def test_two_incr_mod_pow2_different_dst_no_conflict(self):
        """Two INCR_MOD_POW2 writing different LRs must not raise a conflict error."""
        state = _run("""\
SET lr5 cr8;;
SET lr12 cr9;;
SET lr13 cr10;;
INCR_MOD_POW2 lr5 lr12 7; INCR_MOD_POW2 lr6 lr13 8;;
BKPT;;
""",
            cr={8: 0, 9: 1, 10: 128})
        assert state.regfile.get_lr(5) == 1
        assert state.regfile.get_lr(6) == 128


# ============================================================================
# Three LR sub-slots per VLIW (SLOT_COUNT["lr"] == 3)
# ============================================================================


class TestThreeLrSlots:
    def test_three_lr_instructions_parallel(self):
        """Three independent LR ops may execute in the same cycle."""
        state = _run(
            """\
SET lr0 cr8; SET lr1 cr9; SET lr2 cr10;;
BKPT;;
""",
            cr={8: 1, 9: 2, 10: 3},
        )
        assert state.regfile.get_lr(0) == 1
        assert state.regfile.get_lr(1) == 2
        assert state.regfile.get_lr(2) == 3

    def test_fourth_lr_instruction_rejected(self):
        from ipu_as.compound_inst import CompoundInst
        from ipu_as.lark_tree import parse

        ast = parse("SET lr0 cr8; SET lr1 cr9; SET lr2 cr10; SET lr3 cr11;;")
        with pytest.raises(ValueError, match="Too many instructions of type LrInst"):
            CompoundInst(ast[0])

    def test_decode_three_lr_sub_slots(self):
        """Assemble → decode exposes lr_inst_0, lr_inst_1, lr_inst_2."""
        encoded = assemble("SET lr4 cr8; SET lr5 cr9; SET lr6 cr10;;\nBKPT;;")
        assert len(encoded) == 2
        d = decode_instruction_word(encoded[0])
        assert d["lr_inst_0_token_0_lr_inst_opcode"] == 0  # set
        assert d["lr_inst_0_token_2_lr_reg_field"] == 4   # reg = lr4
        assert d["lr_inst_0_token_1_add_sub_src_b_field"] == 8  # src = cr8
        assert d["lr_inst_0_token_3_lr_reg_field"] == 0   # unused field (default lr0)
        assert d["lr_inst_1_token_2_lr_reg_field"] == 5   # reg = lr5
        assert d["lr_inst_1_token_1_add_sub_src_b_field"] == 9   # src = cr9
        assert d["lr_inst_1_token_3_lr_reg_field"] == 0   # unused field (default lr0)
        assert d["lr_inst_2_token_2_lr_reg_field"] == 6   # reg = lr6
        assert d["lr_inst_2_token_1_add_sub_src_b_field"] == 10  # src = cr10
        assert d["lr_inst_2_token_3_lr_reg_field"] == 0   # unused field (default lr0)

    def test_decode_add_imm_operand_field(self):
        """``add`` third operand uses AddSubSrcBField; IMM5 encodes as 32 + value."""
        encoded = assemble("ADD lr2 lr1 7;; BKPT;;")
        d = decode_instruction_word(encoded[0])
        assert d["lr_inst_0_token_0_lr_inst_opcode"] == 1  # add
        assert d["lr_inst_0_token_2_lr_reg_field"] == 2   # dest = lr2
        assert d["lr_inst_0_token_3_lr_reg_field"] == 1   # src_a = lr1
        assert d["lr_inst_0_token_1_add_sub_src_b_field"] == 32 + 7  # src_b = IMM5(7)


# ============================================================================
# Memory Operations
# ============================================================================


class TestMemoryOperations:
    def test_load_from_memory(self):
        test_data = bytes(range(128))
        state = _make_state("""\
SET lr13 cr8;;
LDR_MULT_REG r1 lr13 cr0;;
BKPT;;
""",
            cr={8: 4096})
        state.xmem.write_address(0x1000, test_data)
        run_until_complete(state)

        assert state.regfile.get_lr(13) == 0x1000
        r1_data = state.regfile.get_r(1)
        assert r1_data == bytearray(test_data)

    def test_store_to_memory(self):
        """INT8: r1=all-2, cyclic=all-3, MULT.EE → acc should be 6 per word."""
        r1_data = bytes([2] * 128)
        cyclic_data = bytes([3] * 512)

        state = _make_state("""\
SET lr13 cr8;;
LDR_MULT_REG r1 lr13 cr0;;
SET lr14 cr9;;
SET lr15 cr10;;
LDR_CYCLIC_MULT_REG lr14 cr0 lr15;;
RESET_ACC;;
MULT.EE r1 lr0 0 lr0;
ACC;;
SET lr0 cr11;;
STR_ACC_REG lr0 cr0;;
BKPT;;
""",
            cr={8: 4096, 9: 8192, 10: 0, 11: 12288})
        state.xmem.write_address(0x1000, r1_data)
        state.xmem.write_address(0x2000, cyclic_data)
        run_until_complete(state)

        acc_bytes = state.xmem.read_address(0x3000, 512)
        words = struct.unpack_from("<128i", acc_bytes)
        for i, w in enumerate(words):
            assert w == 6, f"acc word {i} should be 6, got {w}"

    def test_cyclic_register_load(self):
        cyclic_data = bytes([(i * 2) & 0xFF for i in range(128)])
        state = _make_state("""\
SET lr0 cr8;;
SET lr1 cr9;;
LDR_CYCLIC_MULT_REG lr0 cr0 lr1;;
BKPT;;
""",
            cr={8: 20480, 9: 0})
        state.xmem.write_address(0x5000, cyclic_data)
        run_until_complete(state)

        assert state.regfile.get_lr(0) == 0x5000
        loaded = state.regfile.get_r_cyclic_at(0, 128)
        assert loaded == bytearray(cyclic_data)

    def test_mask_register_load(self):
        mask_data = bytes([(i + 1) & 0xFF for i in range(128)])
        state = _make_state("""\
SET lr0 cr8;;
LDR_MULT_MASK_REG lr0 cr0;;
BKPT;;
""",
            cr={8: 24576})
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

        state = _make_state("""\
SET lr0 cr8;;
LDR_MULT_REG r0 lr0 cr0;;
SET lr1 cr9;;
SET lr2 cr10;;
LDR_CYCLIC_MULT_REG lr1 cr0 lr2;;
SET lr3 cr11;;
LDR_MULT_MASK_REG lr3 cr0;;
RESET_ACC;;
SET lr5 cr10;;
SET lr6 cr10;;
MULT.EE r0 lr6 0 lr5;
ACC;;
SET lr9 cr12;;
STR_ACC_REG lr9 cr0;;
BKPT;;
""",
            cr={8: 4096, 9: 8192, 10: 0, 11: 12288, 12: 16384})
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

        state = _make_state("""\
SET lr0 cr8;;
LDR_MULT_REG r0 lr0 cr0;;
SET lr1 cr9;;
SET lr2 cr10;;
LDR_CYCLIC_MULT_REG lr1 cr0 lr2;;
SET lr3 cr11;;
LDR_MULT_MASK_REG lr3 cr0;;
RESET_ACC;;
SET lr5 cr12;;
SET lr6 cr10;;
MULT.EE r0 lr6 1 lr5;
ACC;;
SET lr9 cr13;;
STR_ACC_REG lr9 cr0;;
BKPT;;
""",
            cr={8: 4096, 9: 8192, 10: 0, 11: 12288, 12: 16, 13: 16384})
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
        state = _run("""\
SET lr0 cr8;;
b skip_section;;
SET lr0 cr9;;
skip_section:
SET lr1 cr10;;
BKPT;;
""",
            cr={8: 1, 9: 2, 10: 3})
        assert state.regfile.get_lr(0) == 1
        assert state.regfile.get_lr(1) == 3

    def test_bne(self):
        state = _run("""\
SET lr0 cr8;;
SET lr1 cr9;;
BNE lr0 lr1 not_equal_branch;;
SET lr2 cr10;;
BKPT;;
not_equal_branch:
SET lr2 cr11;;
BKPT;;
""",
            cr={8: 10, 9: 20, 10: 0, 11: 1})
        assert state.regfile.get_lr(2) == 1

    def test_beq(self):
        state = _run("""\
SET lr0 cr8;;
SET lr1 cr8;;
BEQ lr0 lr1 equal_branch;;
SET lr2 cr9;;
BKPT;;
equal_branch:
SET lr2 cr10;;
BKPT;;
""",
            cr={8: 42, 9: 0, 10: 1})
        assert state.regfile.get_lr(2) == 1

    def test_blt(self):
        state = _run("""\
SET lr0 cr8;;
SET lr1 cr9;;
BLT lr0 lr1 less_branch;;
SET lr2 cr10;;
BKPT;;
less_branch:
SET lr2 cr11;;
BKPT;;
""",
            cr={8: 5, 9: 6, 10: 0, 11: 1})
        assert state.regfile.get_lr(2) == 1

    def test_bnz(self):
        state = _run("""\
SET lr0 cr8;;
BNZ lr0 lr0 nonzero_branch;;
SET lr2 cr9;;
BKPT;;
nonzero_branch:
SET lr2 cr8;;
BKPT;;
""",
            cr={8: 1, 9: 0})
        assert state.regfile.get_lr(2) == 1

    def test_bz(self):
        state = _run("""\
SET lr0 cr8;;
BZ lr0 lr0 zero_branch;;
SET lr2 cr8;;
BKPT;;
zero_branch:
SET lr2 cr9;;
BKPT;;
""",
            cr={8: 0, 9: 1})
        assert state.regfile.get_lr(2) == 1

    def test_bne_lr_cr(self):
        """bne branches when LR does not equal a CR constant."""
        state = _make_state("""\
SET lr0 cr8;;
BNE lr0 cr1 bne_cr_branch;;
SET lr2 cr9;;
BKPT;;
bne_cr_branch:
SET lr2 cr10;;
BKPT;;
""",
            cr={8: 10, 9: 0, 10: 1})
        state.regfile.set_cr(1, 20)
        run_until_complete(state)
        assert state.regfile.get_lr(2) == 1

    def test_beq_lr_cr(self):
        """beq branches when LR equals a CR constant."""
        state = _make_state("""\
SET lr0 cr8;;
BEQ lr0 cr3 beq_cr_branch;;
SET lr2 cr9;;
BKPT;;
beq_cr_branch:
SET lr2 cr10;;
BKPT;;
""",
            cr={8: 42, 9: 0, 10: 1})
        state.regfile.set_cr(3, 42)
        run_until_complete(state)
        assert state.regfile.get_lr(2) == 1

    def test_blt_lr_cr(self):
        """blt branches when LR is less than a CR constant."""
        state = _make_state("""\
SET lr0 cr8;;
BLT lr0 cr2 blt_cr_branch;;
SET lr2 cr9;;
BKPT;;
blt_cr_branch:
SET lr2 cr10;;
BKPT;;
""",
            cr={8: 5, 9: 0, 10: 1})
        state.regfile.set_cr(2, 10)
        run_until_complete(state)
        assert state.regfile.get_lr(2) == 1

    def test_bnz_lr_cr(self):
        """bnz branches when test_reg is not zero (CR as base_reg)."""
        state = _make_state("""\
SET lr0 cr8;;
BNZ lr0 cr0 bnz_cr_branch;;
SET lr2 cr9;;
BKPT;;
bnz_cr_branch:
SET lr2 cr10;;
BKPT;;
""",
            cr={8: 5, 9: 0, 10: 1})
        state.regfile.set_cr(0, 0)
        run_until_complete(state)
        assert state.regfile.get_lr(2) == 1

    def test_bz_lr_cr(self):
        """bz branches when test_reg is zero (CR as base_reg)."""
        state = _make_state("""\
SET lr0 cr8;;
BZ lr0 cr0 bz_cr_branch;;
SET lr2 cr8;;
BKPT;;
bz_cr_branch:
SET lr2 cr9;;
BKPT;;
""",
            cr={8: 0, 9: 1})
        state.regfile.set_cr(0, 0)
        run_until_complete(state)
        assert state.regfile.get_lr(2) == 1

    def test_loop_with_cr_limit(self):
        """Loop using bne against a CR constant instead of an LR."""
        state = _make_state("""\
SET lr0 cr8;;
cr_loop_start:
ADD lr0 lr0 1;;
BNE lr0 cr5 cr_loop_start;;
BKPT;;
""",
            cr={8: 0})
        state.regfile.set_cr(5, 10)
        run_until_complete(state, max_cycles=1000)
        assert state.regfile.get_lr(0) == 10

    def test_simple_loop(self):
        state = _run("""\
SET lr0 cr8;;
SET lr1 cr9;;
SET lr2 cr8;;
loop_start:
ADD lr0 lr0 1;;
BNE lr0 lr1 loop_start;;
BKPT;;
""",
            cr={8: 0, 9: 10},
            max_cycles=1000,
        )
        assert state.regfile.get_lr(0) == 10

    def test_BKPT_halts(self):
        state = _run(
            """\
SET lr0 cr10;;
BKPT;;
SET lr0 cr8;;
""",
            cr={8: 0, 9: 10, 10: 99})
        # BKPT sets PC = INST_MEM_SIZE, so `SET lr0 0` is never reached
        assert state.regfile.get_lr(0) == 99

    def test_br_cr(self):
        """br accepts a CR register as the branch target address."""
        state = _make_state("""\
BR cr0;;
SET lr0 cr8;;
BKPT;;
br_cr_target:
SET lr0 cr9;;
BKPT;;
""",
            cr={8: 0, 9: 1})
        from ipu_as.label import ipu_labels
        target_addr = ipu_labels.get_address(
            type("T", (), {"value": "br_cr_target", "line": 0, "column": 0})()
        )
        state.regfile.set_cr(0, target_addr)
        run_until_complete(state)
        assert state.regfile.get_lr(0) == 1


# ============================================================================
# Accumulator
# ============================================================================


class TestAccumulator:
    def test_reset(self):
        state = _make_state("RESET_ACC;;\nBKPT;;")
        # Pre-fill acc words with non-zero
        for i in range(128):
            state.regfile.set_r_acc_word(i, 12345)
        run_until_complete(state)
        for i in range(128):
            assert state.regfile.get_r_acc_word(i) == 0

    def test_acc_add_aaq(self):
        """ACC.ADD_AAQ adds the selected AAQ register (32-bit) to each of the 128 accumulator words."""
        state = _make_state(
            """\
RESET_ACC;;
ACC.ADD_AAQ aaq1;;
BKPT;;
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
        """ACC.FIRST sets r_acc to mult_res without adding previous r_acc."""
        state = _make_state(
            """\
ACC.FIRST;;
BKPT;;
"""
        )
        state.regfile.set_cr(15, DType.INT8)
        # Pre-fill acc with garbage; ACC.FIRST should ignore it
        for i in range(128):
            state.regfile.set_r_acc_word(i, 9999)
        mult_buf = state.regfile.raw("mult_res")
        for i in range(128):
            struct.pack_into("<i", mult_buf, i * 4, 7)
        run_until_complete(state)
        for i in range(128):
            assert state.regfile.get_r_acc_word(i) == 7, f"word {i}: expected 7, got {state.regfile.get_r_acc_word(i)}"

    def test_acc_add_aaq_first(self):
        """ACC.ADD_AAQ.FIRST sets r_acc to mult_res + aaq (no previous sum)."""
        state = _make_state(
            """\
ACC.ADD_AAQ.FIRST aaq2;;
BKPT;;
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
        """ACC.MAX sets r_acc[i] = max(r_acc[i], mult_res[i], aaq_reg)."""
        state = _make_state(
            """\
ACC.MAX aaq1;;
BKPT;;
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
        """ACC.MAX treats register values as signed int32; negative values compare correctly."""
        state = _make_state(
            """\
ACC.MAX aaq0;;
BKPT;;
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
        """ACC.MAX.FIRST sets r_acc[i] = max(mult_res[i], aaq_reg); previous r_acc ignored."""
        state = _make_state(
            """\
ACC.MAX.FIRST aaq0;;
BKPT;;
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
        """ACC.STRIDE with both strides off copies all 128 mult_res words to r_acc from start 0."""
        state = _make_state("""\
SET lr0 cr8;;
ACC.STRIDE 8 off off lr0;;
BKPT;;
""",
            cr={8: 0})
        state.regfile.set_cr(15, DType.INT8)
        mult_buf = state.regfile.raw("mult_res")
        for i in range(128):
            struct.pack_into("<i", mult_buf, i * 4, i)
        run_until_complete(state)
        for i in range(128):
            w = state.regfile.get_r_acc_word(i)
            assert w == i, f"word {i}: expected {i}, got {w}"

    def test_acc_stride_horizontal_no_expand(self):
        """ACC.STRIDE with horizontal on, no expand: take every 2nd column → 64 elements at r_acc[0:64]."""
        state = _make_state("""\
SET lr0 cr8;;
ACC.STRIDE 8 on off lr0;;
BKPT;;
""",
            cr={8: 0})
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
        """ACC.STRIDE with offset: (lr0 % 4)*32 is start index; 64 elements written at r_acc[32:96]."""
        state = _make_state("""\
SET lr0 cr8;;
ACC.STRIDE 8 on off lr0;;
BKPT;;
""",
            cr={8: 1})
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
agg sum value lr1 cr0 aaq0;;
BKPT;;
"""
        )
        state.regfile.set_cr(15, DType.INT8)
        state.regfile.set_lr(1, 128)
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
agg max value lr1 cr0 aaq1;;
BKPT;;
"""
        )
        state.regfile.set_cr(15, DType.INT8)
        state.regfile.set_lr(1, 128)
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
agg max value lr1 cr0 aaq0;;
BKPT;;
"""
        )
        state.regfile.set_cr(15, DType.INT8)
        state.regfile.set_lr(1, 128)
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
agg sum value_cr lr1 cr1 aaq2;;
BKPT;;
"""
        )
        state.regfile.set_cr(15, DType.INT8)
        state.regfile.set_lr(1, 128)
        for i in range(128):
            state.regfile.set_r_acc_word(i, struct.unpack("<I", struct.pack("<i", 1))[0])
        state.regfile.set_cr(1, struct.unpack("<I", struct.pack("<i", 3))[0])
        state.regfile.set_aaq(2, 0)

        run_until_complete(state)
        # sum = 128, 128 * 3 = 384
        assert state.regfile.get_aaq(2) == struct.unpack("<I", struct.pack("<i", 384))[0]

    def test_agg_first_max_ignores_previous_aaq(self):
        """agg.first max: ignores the previous (garbage) AAQ value, takes max of r_acc only."""
        import struct

        state = _make_state(
            """\
agg.first max value lr1 cr0 aaq0;;
BKPT;;
"""
        )
        state.regfile.set_cr(15, DType.INT8)
        state.regfile.set_lr(1, 128)
        for i in range(128):
            state.regfile.set_r_acc_word(i, struct.unpack("<I", struct.pack("<i", 10 + (i % 5)))[0])
        # Set aaq0 to a large "garbage" value that would win against r_acc if included
        state.regfile.set_aaq(0, struct.unpack("<I", struct.pack("<i", 9999))[0])

        run_until_complete(state)
        # Max of r_acc is 14; previous aaq (9999) must be ignored
        assert state.regfile.get_aaq(0) == struct.unpack("<I", struct.pack("<i", 14))[0]

    def test_agg_first_max_selects_correct_max(self):
        """agg.first max: correctly selects the max value from r_acc words."""
        import struct

        state = _make_state(
            """\
agg.first max value lr1 cr0 aaq1;;
BKPT;;
"""
        )
        state.regfile.set_cr(15, DType.INT8)
        state.regfile.set_lr(1, 128)
        for i in range(128):
            state.regfile.set_r_acc_word(i, struct.unpack("<I", struct.pack("<i", 5))[0])
        state.regfile.set_r_acc_word(63, struct.unpack("<I", struct.pack("<i", 77))[0])
        state.regfile.set_aaq(1, struct.unpack("<I", struct.pack("<i", 0))[0])

        run_until_complete(state)
        assert state.regfile.get_aaq(1) == struct.unpack("<I", struct.pack("<i", 77))[0]

    def test_agg_first_sum_same_as_agg_sum(self):
        """agg.first sum: behaves identically to agg sum (previous aaq not involved in sum)."""
        import struct

        state = _make_state(
            """\
agg.first sum value lr1 cr0 aaq2;;
BKPT;;
"""
        )
        state.regfile.set_cr(15, DType.INT8)
        state.regfile.set_lr(1, 128)
        for i in range(128):
            state.regfile.set_r_acc_word(i, struct.unpack("<I", struct.pack("<i", 2))[0])
        state.regfile.set_aaq(2, struct.unpack("<I", struct.pack("<i", 9999))[0])

        run_until_complete(state)
        # Sum of 128 twos = 256; previous aaq value irrelevant
        assert state.regfile.get_aaq(2) == struct.unpack("<I", struct.pack("<i", 256))[0]

    def test_agg_sum_masks_inactive_lanes(self):
        """agg sum: only first valid_elements r_acc words contribute (tail masked out)."""
        import struct

        state = _make_state(
            """\
agg sum value lr1 cr0 aaq0;;
BKPT;;
"""
        )
        state.regfile.set_cr(15, DType.INT8)
        state.regfile.set_lr(1, 3)
        for i in range(128):
            v = 10 if i < 3 else 1000
            state.regfile.set_r_acc_word(i, struct.unpack("<I", struct.pack("<i", v))[0])
        state.regfile.set_aaq(0, 0)
        run_until_complete(state)
        assert state.regfile.get_aaq(0) == struct.unpack("<I", struct.pack("<i", 30))[0]

    def test_agg_sum_valid_elements_from_cr(self):
        """agg sum: valid_elements may be read from a CR register (LcrIdx encoding)."""
        import struct

        state = _make_state(
            """\
agg sum value cr2 cr0 aaq0;;
BKPT;;
"""
        )
        state.regfile.set_cr(15, DType.INT8)
        state.regfile.set_cr(2, 100)
        for i in range(128):
            v = 1 if i < 100 else 500
            state.regfile.set_r_acc_word(i, struct.unpack("<I", struct.pack("<i", v))[0])
        state.regfile.set_aaq(0, 0)
        run_until_complete(state)
        assert state.regfile.get_aaq(0) == struct.unpack("<I", struct.pack("<i", 100))[0]

    def test_agg_max_masks_tail_large_values(self):
        """agg max: masked-out lanes cannot beat AAQ; feedback still applies over active set."""
        import struct

        state = _make_state(
            """\
agg max value lr1 cr0 aaq0;;
BKPT;;
"""
        )
        state.regfile.set_cr(15, DType.INT8)
        state.regfile.set_lr(1, 100)
        for i in range(128):
            v = 5 if i < 100 else 9999
            state.regfile.set_r_acc_word(i, struct.unpack("<I", struct.pack("<i", v))[0])
        state.regfile.set_aaq(0, struct.unpack("<I", struct.pack("<i", 50))[0])
        run_until_complete(state)
        assert state.regfile.get_aaq(0) == struct.unpack("<I", struct.pack("<i", 50))[0]

    def test_agg_first_max_masks_tail(self):
        """agg.first max: only active prefix participates; tail spikes ignored."""
        import struct

        state = _make_state(
            """\
agg.first max value lr1 cr0 aaq0;;
BKPT;;
"""
        )
        state.regfile.set_cr(15, DType.INT8)
        state.regfile.set_lr(1, 50)
        for i in range(128):
            v = 3 if i < 50 else 200
            state.regfile.set_r_acc_word(i, struct.unpack("<I", struct.pack("<i", v))[0])
        state.regfile.set_aaq(0, struct.unpack("<I", struct.pack("<i", 0))[0])
        run_until_complete(state)
        assert state.regfile.get_aaq(0) == struct.unpack("<I", struct.pack("<i", 3))[0]


# ============================================================================


class TestProgramCounter:
    def test_pc_advances(self):
        from ipu_emu.execute import execute_next_instruction

        state = _make_state("""\
SET lr0 cr8;;
SET lr1 cr9;;
BKPT;;
""",
            cr={8: 100, 9: 200})
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

        state = _make_state("""\
SET lr0 cr8;;
BREAK;;
SET lr0 cr9;;
BKPT;;
""",
            cr={8: 1, 9: 2})
        r = execute_next_instruction(state)
        assert r == BreakResult.CONTINUE
        assert state.regfile.get_lr(0) == 1

        r = execute_next_instruction(state)
        assert r == BreakResult.BREAK

    def test_break_ifeq(self):
        from ipu_emu.execute import execute_next_instruction

        state = _make_state("""\
SET lr5 cr8;;
BREAK.IFEQ lr5 42;;
BKPT;;
""",
            cr={8: 42})
        execute_next_instruction(state)  # SET lr5 42
        assert state.regfile.get_lr(5) == 42

        r = execute_next_instruction(state)
        assert r == BreakResult.BREAK

    def test_break_ifeq_no_match(self):
        from ipu_emu.execute import execute_next_instruction

        state = _make_state("""\
SET lr5 cr8;;
BREAK.IFEQ lr5 42;;
BKPT;;
""",
            cr={8: 10})
        execute_next_instruction(state)
        r = execute_next_instruction(state)
        assert r == BreakResult.CONTINUE

    def test_run_with_debug_step(self):
        actions = iter([DebugAction.STEP, DebugAction.CONTINUE])

        def callback(state, cycle):
            return next(actions, DebugAction.CONTINUE)

        state = _make_state("""\
BREAK;;
SET lr0 cr8;;
BKPT;;
""",
            cr={8: 1})
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
        encoded = assemble("SET lr13 cr8;;\nBKPT;;")
        assert len(encoded) == 2
        d = decode_instruction_word(encoded[0])
        # LR opcode should be 'set' = index 0
        assert d["lr_inst_0_token_0_lr_inst_opcode"] == 0  # set
        assert d["lr_inst_0_token_2_lr_reg_field"] == 13  # reg = lr13
        assert d["lr_inst_0_token_1_add_sub_src_b_field"] == 8  # src = cr8


# ============================================================================
# FP8 multiply
# ============================================================================


class TestFp8:
    def test_fp8_e4m3_mult_ee(self):
        """FP8 E4 (e4m3): 1.0 × 2.0 → 2.0 for every element."""
        from ipu_emu.ipu_math import _float32_to_fp8_scalar

        fp_one_byte = _float32_to_fp8_scalar(1.0, 4)
        fp_two_byte = _float32_to_fp8_scalar(2.0, 4)

        r0_data = bytes([fp_one_byte] * 128)
        cyclic_data = bytes([fp_two_byte] * 512)

        state = _make_state("""\
SET lr0 cr8;;
LDR_MULT_REG r0 lr0 cr0;;
SET lr1 cr9;;
SET lr2 cr10;;
LDR_CYCLIC_MULT_REG lr1 cr0 lr2;;
RESET_ACC;;
SET lr5 cr10;;
SET lr6 cr10;;
MULT.EE r0 lr6 0 lr5;
ACC;;
BKPT;;
""",
            cr={8: 4096, 9: 8192, 10: 0})
        # Set dtype to E4 (fp8_e4)
        state.set_cr_dtype(DType.E4)
        state.xmem.write_address(0x1000, r0_data)
        state.xmem.write_address(0x2000, cyclic_data)
        run_until_complete(state)

        # Read back acc as floats
        acc_raw = state.regfile.raw("r_acc")
        for i in range(128):
            val = struct.unpack_from("<f", acc_raw, i * 4)[0]
            assert abs(val - 2.0) < 0.01, f"acc word {i}: expected 2.0, got {val}"


# ============================================================================
# MULT.VE.CR and MULT.VE.AAQ
# ============================================================================


class TestMultVeCrAaq:
    """Tests for MULT.VE.CR and MULT.VE.AAQ instructions."""

    # ------------------------------------------------------------------
    # MULT.VE.CR
    # ------------------------------------------------------------------

    def test_mult_ve_cr_int8(self):
        """MULT.VE.CR INT8: scalar from CR × RC elements."""
        # CR scalar byte = 3, RC elements = all 2 → each result = 3*2 = 6
        cyclic_data = bytes([2] * 512)

        state = _make_state("""\
SET lr0 cr8;;
SET lr1 cr9;;
LDR_CYCLIC_MULT_REG lr0 cr0 lr1;;
RESET_ACC;;
SET lr2 cr9;;
SET lr4 cr9;;
MULT.VE.CR lr2 0 lr4 cr1;
ACC;;
BKPT;;
""",
            cr={8: 4096, 9: 0})
        state.regfile.set_cr(15, DType.INT8)
        state.regfile.set_cr(1, 3)  # scalar = low byte = 3
        state.xmem.write_address(0x1000, cyclic_data)
        run_until_complete(state)

        acc_raw = state.regfile.raw("r_acc")
        for i in range(128):
            val = struct.unpack_from("<i", acc_raw, i * 4)[0]
            assert val == 6, f"acc word {i}: expected 6, got {val}"

    def test_mult_ve_cr_negative_int8(self):
        """MULT.VE.CR INT8: signed negative scalar × positive RC elements."""
        # CR scalar byte = 0xFE = -2 (signed int8), RC elements = 5 → result = -10
        cyclic_data = bytes([5] * 512)

        state = _make_state("""\
SET lr0 cr8;;
SET lr1 cr9;;
LDR_CYCLIC_MULT_REG lr0 cr0 lr1;;
RESET_ACC;;
SET lr2 cr9;;
SET lr4 cr9;;
MULT.VE.CR lr2 0 lr4 cr2;
ACC;;
BKPT;;
""",
            cr={8: 4096, 9: 0})
        state.regfile.set_cr(15, DType.INT8)
        state.regfile.set_cr(2, 0xFE)  # low byte 0xFE = int8(-2)
        state.xmem.write_address(0x1000, cyclic_data)
        run_until_complete(state)

        acc_raw = state.regfile.raw("r_acc")
        for i in range(128):
            val = struct.unpack_from("<i", acc_raw, i * 4)[0]
            assert val == -10, f"acc word {i}: expected -10, got {val}"

    def test_mult_ve_cr_boundary_padding(self):
        """MULT.VE.CR: elements beyond RC boundary (512 bytes) are padded with int8 1."""
        # cyclic_offset = 450, so first 62 bytes come from RC, remaining 66 are padded with 1
        rc_fill = 4  # RC filled with 4
        scalar = 7
        pad_start = 62  # 512 - 450 = 62 elements in bounds

        state = _make_state("""\
RESET_ACC;;
SET lr2 cr8;;
SET lr4 cr9;;
MULT.VE.CR lr2 0 lr4 cr3;
ACC;;
BKPT;;
""",
            cr={8: 450, 9: 0})
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
        """MULT.VE.CR fp8_e4: scalar 1.0 × RC elements 1.0 → result 1.0 each."""
        from ipu_emu.ipu_math import _float32_to_fp8_scalar

        one_fp8 = _float32_to_fp8_scalar(1.0, 4)
        cyclic_data = bytes([one_fp8] * 512)

        state = _make_state("""\
SET lr0 cr8;;
SET lr1 cr9;;
LDR_CYCLIC_MULT_REG lr0 cr0 lr1;;
RESET_ACC;;
SET lr2 cr9;;
SET lr4 cr9;;
MULT.VE.CR lr2 0 lr4 cr5;
ACC;;
BKPT;;
""",
            cr={8: 4096, 9: 0})
        state.regfile.set_cr(15, DType.E4)
        state.regfile.set_cr(5, one_fp8)  # scalar = 1.0 in fp8_e4
        state.xmem.write_address(0x1000, cyclic_data)
        run_until_complete(state)

        acc_raw = state.regfile.raw("r_acc")
        for i in range(128):
            val = struct.unpack_from("<f", acc_raw, i * 4)[0]
            assert abs(val - 1.0) < 0.01, f"acc word {i}: expected 1.0, got {val}"

    def test_mult_ve_cr_boundary_padding_fp8e5m2(self):
        """MULT.VE.CR fp8_e5: boundary elements padded with FP8 1.0."""
        from ipu_emu.ipu_math import _float32_to_fp8_scalar

        two_fp8 = _float32_to_fp8_scalar(2.0, 5)
        scalar_fp8 = _float32_to_fp8_scalar(3.0, 5)

        state = _make_state("""\
RESET_ACC;;
SET lr2 cr8;;
SET lr4 cr9;;
MULT.VE.CR lr2 0 lr4 cr6;
ACC;;
BKPT;;
""",
            cr={8: 500, 9: 0})
        state.regfile.set_cr(15, DType.E5)
        state.regfile.set_cr(6, scalar_fp8)  # scalar = 3.0
        # Fill the full 512-byte cyclic register directly with 2.0 in fp8_e5
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
    # MULT.VE.AAQ
    # ------------------------------------------------------------------

    def test_mult_ve_aaq_int8(self):
        """MULT.VE.AAQ INT8: scalar from AAQ × RC elements."""
        # AAQ scalar byte = 5, RC elements = all 3 → each result = 15
        cyclic_data = bytes([3] * 512)

        state = _make_state("""\
SET lr0 cr8;;
SET lr1 cr9;;
LDR_CYCLIC_MULT_REG lr0 cr0 lr1;;
RESET_ACC;;
SET lr2 cr9;;
SET lr4 cr9;;
MULT.VE.AAQ lr2 0 lr4 aaq0;
ACC;;
BKPT;;
""",
            cr={8: 4096, 9: 0})
        state.regfile.set_cr(15, DType.INT8)
        state.regfile.set_aaq(0, 5)  # low byte = 5
        state.xmem.write_address(0x1000, cyclic_data)
        run_until_complete(state)

        acc_raw = state.regfile.raw("r_acc")
        for i in range(128):
            val = struct.unpack_from("<i", acc_raw, i * 4)[0]
            assert val == 15, f"acc word {i}: expected 15, got {val}"

    def test_mult_ve_aaq_boundary_padding(self):
        """MULT.VE.AAQ: elements beyond RC boundary are padded with int8 1."""
        scalar = 2
        in_bounds = 112  # offset=400, 512-400=112 in bounds, 16 padded

        state = _make_state("""\
RESET_ACC;;
SET lr2 cr8;;
SET lr4 cr9;;
MULT.VE.AAQ lr2 0 lr4 aaq1;
ACC;;
BKPT;;
""",
            cr={8: 400, 9: 0})
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
        """MULT.VE.AAQ: when cyclic_offset+128 <= 512, no padding applied."""
        cyclic_data = bytes([7] * 512)
        scalar = 3

        state = _make_state("""\
SET lr0 cr8;;
SET lr1 cr9;;
LDR_CYCLIC_MULT_REG lr0 cr0 lr1;;
RESET_ACC;;
SET lr2 cr9;;
SET lr4 cr9;;
MULT.VE.AAQ lr2 0 lr4 aaq2;
ACC;;
BKPT;;
""",
            cr={8: 4096, 9: 0})
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
        """MULT.EE still works correctly after adding new mult variants."""
        r0_data = bytes([4] * 128)
        cyclic_data = bytes([5] * 512)

        state = _make_state("""\
SET lr0 cr8;;
LDR_MULT_REG r0 lr0 cr0;;
SET lr1 cr9;;
SET lr2 cr10;;
LDR_CYCLIC_MULT_REG lr1 cr0 lr2;;
RESET_ACC;;
MULT.EE r0 lr2 0 lr2;
ACC;;
BKPT;;
""",
            cr={8: 4096, 9: 8192, 10: 0})
        state.regfile.set_cr(15, DType.INT8)
        state.xmem.write_address(0x1000, r0_data)
        state.xmem.write_address(0x2000, cyclic_data)
        run_until_complete(state)

        acc_raw = state.regfile.raw("r_acc")
        for i in range(128):
            val = struct.unpack_from("<i", acc_raw, i * 4)[0]
            assert val == 20, f"acc word {i}: expected 20, got {val}"

    def test_backward_compat_mult_ve(self):
        """MULT.VE.CYCLIC still works correctly after adding new mult variants."""
        cyclic_data = bytes([6] * 512)
        r0_data = bytes([0] * 128)
        r0_data = bytearray(r0_data)
        r0_data[0] = 3  # fixed_idx=0 → r0[0] = 3

        state = _make_state("""\
SET lr0 cr8;;
LDR_MULT_REG r0 lr0 cr0;;
SET lr1 cr9;;
SET lr2 cr10;;
LDR_CYCLIC_MULT_REG lr1 cr0 lr2;;
RESET_ACC;;
MULT.VE.CYCLIC lr2 0 lr2 lr2;
ACC;;
BKPT;;
""",
            cr={8: 4096, 9: 8192, 10: 0})
        state.regfile.set_cr(15, DType.INT8)
        state.xmem.write_address(0x1000, bytes(r0_data))
        state.xmem.write_address(0x2000, cyclic_data)
        run_until_complete(state)

        acc_raw = state.regfile.raw("r_acc")
        for i in range(128):
            val = struct.unpack_from("<i", acc_raw, i * 4)[0]
            assert val == 18, f"acc word {i}: expected 18, got {val}"

    def test_mult_ve_cyclic_wrap_at_rc_boundary(self):
        """MULT.VE.CYCLIC: RC indices wrap modulo 512."""
        # cyclic_offset = 450, so without wrap we'd read past 512; with wrap, bytes
        # 450..511 then 0..65 of RC are used — all rc_fill.
        rc_fill = 4
        scalar = 5

        r0_data = bytearray(128)
        r0_data[0] = scalar  # fixed_idx=0 → r0[0] = 5

        state = _make_state("""\
SET lr0 cr8;;
LDR_MULT_REG r0 lr0 cr0;;
RESET_ACC;;
SET lr2 cr9;;
SET lr4 cr10;;
SET lr5 cr10;;
MULT.VE.CYCLIC lr2 0 lr4 lr5;
ACC;;
BKPT;;
""",
            cr={8: 4096, 9: 450, 10: 0})
        state.regfile.set_cr(15, DType.INT8)
        state.xmem.write_address(0x1000, bytes(r0_data))
        state.regfile.set_r_cyclic_at(0, bytes([rc_fill] * 512))
        run_until_complete(state)

        acc_raw = state.regfile.raw("r_acc")
        for i in range(128):
            val = struct.unpack_from("<i", acc_raw, i * 4)[0]
            assert val == scalar * rc_fill, f"word {i}: expected {scalar * rc_fill}, got {val}"

    def test_mult_ve_padded_boundary(self):
        """MULT.VE.PADDED: elements past RC byte 511 use dtype 1."""
        rc_fill = 4
        scalar = 5
        pad_start = 62  # 512 - 450 = 62 elements in bounds before padding

        r0_data = bytearray(128)
        r0_data[0] = scalar

        state = _make_state("""\
SET lr0 cr8;;
LDR_MULT_REG r0 lr0 cr0;;
RESET_ACC;;
SET lr2 cr9;;
SET lr4 cr10;;
SET lr5 cr10;;
MULT.VE.PADDED lr2 0 lr4 lr5;
ACC;;
BKPT;;
""",
            cr={8: 4096, 9: 450, 10: 0})
        state.regfile.set_cr(15, DType.INT8)
        state.xmem.write_address(0x1000, bytes(r0_data))
        state.regfile.set_r_cyclic_at(0, bytes([rc_fill] * 512))
        run_until_complete(state)

        acc_raw = state.regfile.raw("r_acc")
        for i in range(pad_start):
            val = struct.unpack_from("<i", acc_raw, i * 4)[0]
            assert val == scalar * rc_fill, f"word {i}: expected {scalar * rc_fill}, got {val}"
        for i in range(pad_start, 128):
            val = struct.unpack_from("<i", acc_raw, i * 4)[0]
            assert val == scalar * 1, f"word {i} (padded): expected {scalar}, got {val}"

    def test_mult_ve_r1_scalar(self):
        """MULT.VE.CYCLIC: fixed_idx in [128, 255] addresses R1[fixed_idx - 128] instead of R0."""
        r0_data = bytearray(128)  # all zeros — must not be picked
        r1_data = bytearray(128)
        r1_data[0] = 7  # fixed_idx=128 → r1[0] = 7
        cyclic_data = bytes([4] * 512)

        state = _make_state("""\
SET lr0 cr8;;
LDR_MULT_REG r0 lr0 cr0;;
SET lr0 cr9;;
LDR_MULT_REG r1 lr0 cr0;;
SET lr1 cr10;;
SET lr2 cr11;;
LDR_CYCLIC_MULT_REG lr1 cr0 lr2;;
RESET_ACC;;
SET lr3 cr12;;
MULT.VE.CYCLIC lr2 0 lr2 lr3;
ACC;;
BKPT;;
""",
            cr={8: 4096, 9: 4352, 10: 8192, 11: 0, 12: 128})
        state.regfile.set_cr(15, DType.INT8)
        state.xmem.write_address(0x1000, bytes(r0_data))
        state.xmem.write_address(0x1100, bytes(r1_data))
        state.xmem.write_address(0x2000, cyclic_data)
        run_until_complete(state)

        acc_raw = state.regfile.raw("r_acc")
        for i in range(128):
            val = struct.unpack_from("<i", acc_raw, i * 4)[0]
            assert val == 28, f"acc word {i}: expected 28 (r1[0]=7 × cyclic[i]=4), got {val}"


class TestMultEeRr:
    """MULT.EE.RR — multi-element execution (MEE): r0-by-r0 or r1-by-r1."""

    def test_mult_ee_rr_r0_by_r0(self):
        """MEE mode R0: each lane multiplied by itself (4 × 4 = 16)."""
        r0_data = bytes([4] * 128)

        state = _make_state("""\
SET lr0 cr8;;
LDR_MULT_REG r0 lr0 cr0;;
RESET_ACC;;
SET lr5 cr9;;
MULT.EE.RR r0 0 lr5;
ACC;;
BKPT;;
""",
            cr={8: 4096, 9: 0})
        state.regfile.set_cr(15, DType.INT8)
        state.xmem.write_address(0x1000, r0_data)
        run_until_complete(state)

        acc_raw = state.regfile.raw("r_acc")
        for i in range(128):
            val = struct.unpack_from("<i", acc_raw, i * 4)[0]
            assert val == 16, f"acc word {i}: expected 16 (4×4), got {val}"

    def test_mult_ee_rr_r1_by_r1(self):
        """MEE mode R1 squares r1 (3 × 3 = 9); r0 holds a decoy value."""
        r0_data = bytes([9] * 128)   # decoy — must be ignored when ra=R1
        r1_data = bytes([3] * 128)

        state = _make_state("""\
SET lr0 cr8;;
LDR_MULT_REG r0 lr0 cr0;;
SET lr1 cr9;;
LDR_MULT_REG r1 lr1 cr0;;
RESET_ACC;;
SET lr5 cr10;;
MULT.EE.RR r1 0 lr5;
ACC;;
BKPT;;
""",
            cr={8: 4096, 9: 4352, 10: 0})
        state.regfile.set_cr(15, DType.INT8)
        state.xmem.write_address(0x1000, r0_data)
        state.xmem.write_address(0x1100, r1_data)
        run_until_complete(state)

        acc_raw = state.regfile.raw("r_acc")
        for i in range(128):
            val = struct.unpack_from("<i", acc_raw, i * 4)[0]
            assert val == 9, f"acc word {i}: expected 9 (r1 3×3, not r0), got {val}"

    def test_mult_ee_rr_mask_zeroes_lanes(self):
        """mask_offset/mask_shift still gate lanes (first 64 masked → 0)."""
        r0_data = bytes([3] * 128)   # squared → 9
        mask_data = bytearray(128)
        for i in range(8):           # 64 bits set → first 64 lanes masked
            mask_data[i] = 0xFF

        state = _make_state("""\
SET lr0 cr8;;
LDR_MULT_REG r0 lr0 cr0;;
SET lr3 cr9;;
LDR_MULT_MASK_REG lr3 cr0;;
RESET_ACC;;
SET lr5 cr10;;
MULT.EE.RR r0 0 lr5;
ACC;;
BKPT;;
""",
            cr={8: 4096, 9: 8192, 10: 0})
        state.regfile.set_cr(15, DType.INT8)
        state.xmem.write_address(0x1000, r0_data)
        state.xmem.write_address(0x2000, bytes(mask_data))
        run_until_complete(state)

        acc_raw = state.regfile.raw("r_acc")
        for i in range(64):
            val = struct.unpack_from("<i", acc_raw, i * 4)[0]
            assert val == 0, f"word {i} should be masked to 0, got {val}"
        for i in range(64, 128):
            val = struct.unpack_from("<i", acc_raw, i * 4)[0]
            assert val == 9, f"word {i}: expected 9 (3×3), got {val}"


# ============================================================================
# AAQ Quantization (aaq instruction + str_post_aaq_reg)
# ============================================================================


class TestAaqQuantize:
    """Tests for the aaq quantization instruction and STR_POST_AAQ_REG."""

    def _set_acc_words(self, state: IpuState, values: list[int]) -> None:
        """Write a list of 128 signed INT32 values into r_acc."""
        assert len(values) == 128
        buf = bytearray(512)
        struct.pack_into("<128i", buf, 0, *values)
        state.regfile.set_r_acc_bytes(buf)

    def test_aaq_basic_truncation(self):
        """Values that fit in int8 after >> 24: e.g. 1 << 24 → byte 1."""
        state = IpuState()
        state.regfile.set_cr(15, DType.INT8)
        self._set_acc_words(state, [i << 24 for i in range(128)])

        encoded = assemble("aaq;;\nBKPT;;")
        from ipu_emu.execute import decode_instruction_word
        from ipu_emu.emulator import load_program, run_until_complete
        decoded = [decode_instruction_word(w) for w in encoded]
        load_program(state, decoded)
        run_until_complete(state)

        result = state.regfile.get_post_aaq_reg()
        for i in range(128):
            expected = i if i < 128 else i - 256
            assert result[i] == (expected & 0xFF), f"byte {i}: expected {expected & 0xFF}, got {result[i]}"
        assert result[128:] == bytearray(384), "tail of POST_AAQ_REG should be cleared"

    def test_aaq_all_zeros(self):
        """All-zero accumulator quantizes to all-zero bytes."""
        state = IpuState()
        state.regfile.set_cr(15, DType.INT8)
        self._set_acc_words(state, [0] * 128)

        encoded = assemble("aaq;;\nBKPT;;")
        from ipu_emu.execute import decode_instruction_word
        from ipu_emu.emulator import load_program, run_until_complete
        decoded = [decode_instruction_word(w) for w in encoded]
        load_program(state, decoded)
        run_until_complete(state)

        assert state.regfile.get_post_aaq_reg() == bytearray(512)

    def test_aaq_positive_clamp(self):
        """Large positive values clamp to 127 after truncation."""
        # 0x7FFFFFFF >> 24 = 127, which is already at the boundary — no clamp needed.
        # Use 0x7F000000 (127 << 24) and 0x80000000 (-128 << 24 in signed) for boundary.
        state = IpuState()
        state.regfile.set_cr(15, DType.INT8)
        values = [0x7F000000] * 64 + [0x7FFFFFFF] * 64
        self._set_acc_words(state, values)

        encoded = assemble("aaq;;\nBKPT;;")
        from ipu_emu.execute import decode_instruction_word
        from ipu_emu.emulator import load_program, run_until_complete
        decoded = [decode_instruction_word(w) for w in encoded]
        load_program(state, decoded)
        run_until_complete(state)

        result = state.regfile.get_post_aaq_reg()
        for i in range(64):
            assert result[i] == 127, f"byte {i}: expected 127, got {result[i]}"
        for i in range(64, 128):
            assert result[i] == 127, f"byte {i}: expected 127, got {result[i]}"
        assert result[128:] == bytearray(384)

    def test_aaq_negative_values(self):
        """Negative accumulator values truncate correctly."""
        # -1 << 24 = 0xFF000000 (signed int32: -16777216); >> 24 = -1 → 0xFF as byte
        state = IpuState()
        state.regfile.set_cr(15, DType.INT8)
        self._set_acc_words(state, [(-1) << 24] * 128)

        encoded = assemble("aaq;;\nBKPT;;")
        from ipu_emu.execute import decode_instruction_word
        from ipu_emu.emulator import load_program, run_until_complete
        decoded = [decode_instruction_word(w) for w in encoded]
        load_program(state, decoded)
        run_until_complete(state)

        result = state.regfile.get_post_aaq_reg()
        for i in range(128):
            assert result[i] == 0xFF, f"byte {i}: expected 0xFF (-1), got {result[i]}"
        assert result[128:] == bytearray(384)

    def test_aaq_requires_int8_mode(self):
        """aaq raises EmulatorError when not in INT8 mode."""
        from ipu_emu.ipu import EmulatorError
        state = _make_state("aaq;;\nBKPT;;")
        state.set_cr_dtype(DType.E4)
        with pytest.raises(EmulatorError, match="INT8 mode"):
            run_until_complete(state)

    def test_str_post_aaq_reg_writes_post_aaq_reg_512_to_xmem(self):
        """STR_POST_AAQ_REG stores POST_AAQ_REG (512 B) to XMEM; emulator syncs from R_ACC."""
        state = IpuState()
        state.regfile.set_cr(15, DType.INT8)
        self._set_acc_words(state, [i << 24 for i in range(128)])
        state.regfile.set_cr(8, 0x4000)

        encoded = assemble(
            """\
SET lr0 cr8;;
str_post_aaq_reg lr0 cr0;;
BKPT;;
"""
        )
        from ipu_emu.execute import decode_instruction_word
        from ipu_emu.emulator import load_program, run_until_complete
        decoded = [decode_instruction_word(w) for w in encoded]
        load_program(state, decoded)
        run_until_complete(state)

        stored = state.xmem.read_address(0x4000, 512)
        assert stored == state.regfile.get_r_acc_bytes()
        assert stored == state.regfile.get_post_aaq_reg()

    def test_STR_ACC_REG_emits_warning(self):
        """STR_ACC_REG emits a UserWarning about being debug-only."""
        state = IpuState()
        state.regfile.set_cr(15, DType.INT8)

        encoded = assemble("STR_ACC_REG lr0 cr0;;\nBKPT;;")
        from ipu_emu.execute import decode_instruction_word
        from ipu_emu.emulator import load_program, run_until_complete
        decoded = [decode_instruction_word(w) for w in encoded]
        load_program(state, decoded)

        with pytest.warns(UserWarning, match="DEBUG ONLY"):
            run_until_complete(state)


# ============================================================================
# ACTIVATE (element-wise r_acc activations, issue #77)
# ============================================================================


class TestActivate:
    """ACTIVATE applies ipu_common.activations to the first ``valid_elements`` r_acc lanes."""

    def test_activate_relu_int32(self):
        state = _make_state(
            """\
ACTIVATE lr0 relu;;
BKPT;;
"""
        )
        state.regfile.set_cr(15, DType.INT8)
        state.regfile.set_lr(0, 128)
        state.regfile.set_r_acc_word(
            0, struct.unpack("<I", struct.pack("<i", -9))[0]
        )
        run_until_complete(state)
        w0 = struct.unpack("<i", struct.pack("<I", state.regfile.get_r_acc_word(0)))[0]
        assert w0 == 0

    def test_activate_masks_inactive_lanes(self):
        state = _make_state(
            """\
ACTIVATE lr0 relu;;
BKPT;;
"""
        )
        state.regfile.set_cr(15, DType.INT8)
        state.regfile.set_lr(0, 2)
        state.regfile.set_r_acc_word(
            0, struct.unpack("<I", struct.pack("<i", -5))[0]
        )
        state.regfile.set_r_acc_word(
            1, struct.unpack("<I", struct.pack("<i", -3))[0]
        )
        sentinel = struct.unpack("<I", struct.pack("<i", 99_999))[0]
        for i in range(2, 128):
            state.regfile.set_r_acc_word(i, sentinel)
        run_until_complete(state)
        assert (
            struct.unpack("<i", struct.pack("<I", state.regfile.get_r_acc_word(0)))[0]
            == 0
        )
        assert (
            struct.unpack("<i", struct.pack("<I", state.regfile.get_r_acc_word(1)))[0]
            == 0
        )
        for i in range(2, 128):
            assert state.regfile.get_r_acc_word(i) == sentinel

    def test_activate_identity_keyword_is_noop(self):
        state = _make_state(
            """\
ACTIVATE lr0 identity;;
BKPT;;
"""
        )
        state.regfile.set_cr(15, DType.INT8)
        state.regfile.set_lr(0, 1)
        raw = struct.unpack("<I", struct.pack("<i", -42))[0]
        state.regfile.set_r_acc_word(0, raw)
        run_until_complete(state)
        assert state.regfile.get_r_acc_word(0) == raw

    def test_activate_sigmoid_float_lane(self):
        state = _make_state(
            """\
ACTIVATE lr0 sigmoid;;
BKPT;;
"""
        )
        state.regfile.set_cr(15, DType.E4)
        state.regfile.set_lr(0, 1)
        state.regfile.set_r_acc_word(
            0, struct.unpack("<I", struct.pack("<f", 0.0))[0]
        )
        run_until_complete(state)
        out = struct.unpack(
            "<f", struct.pack("<I", state.regfile.get_r_acc_word(0))
        )[0]
        assert abs(out - 0.5) < 1e-6

    def test_activate_matches_reference_all_ids_float(self):
        x = 0.25
        for fid, name in enumerate(ACTIVATION_FN_NAMES):
            state = _make_state(
                f"""\
ACTIVATE lr0 {name};;
BKPT;;
"""
            )
            state.regfile.set_cr(15, DType.E4)
            state.regfile.set_lr(0, 1)
            state.regfile.set_r_acc_word(
                0, struct.unpack("<I", struct.pack("<f", x))[0]
            )
            run_until_complete(state)
            got = struct.unpack(
                "<f", struct.pack("<I", state.regfile.get_r_acc_word(0))
            )[0]
            exp = apply_activation(fid, x)
            assert abs(got - exp) < 1e-5, f"id={fid} got={got} exp={exp}"

    def test_activate_exp2_float(self):
        state = _make_state(
            """\
ACTIVATE lr0 exp2;;
BKPT;;
"""
        )
        state.regfile.set_cr(15, DType.E4)
        state.regfile.set_lr(0, 1)
        state.regfile.set_r_acc_word(
            0, struct.unpack("<I", struct.pack("<f", 3.0))[0]
        )
        run_until_complete(state)
        out = struct.unpack(
            "<f", struct.pack("<I", state.regfile.get_r_acc_word(0))
        )[0]
        assert abs(out - 8.0) < 1e-5

    def test_activate_gelu_float(self):
        state = _make_state(
            """\
ACTIVATE lr0 gelu;;
BKPT;;
"""
        )
        state.regfile.set_cr(15, DType.E4)
        state.regfile.set_lr(0, 1)
        x = 1.0
        state.regfile.set_r_acc_word(
            0, struct.unpack("<I", struct.pack("<f", x))[0]
        )
        run_until_complete(state)
        out = struct.unpack(
            "<f", struct.pack("<I", state.regfile.get_r_acc_word(0))
        )[0]
        assert abs(out - apply_activation(6, x)) < 1e-5

    def test_activate_swish_alias_matches_silu(self):
        state = _make_state(
            """\
ACTIVATE lr0 swish;;
BKPT;;
"""
        )
        state.regfile.set_cr(15, DType.E4)
        state.regfile.set_lr(0, 1)
        x = 0.5
        state.regfile.set_r_acc_word(
            0, struct.unpack("<I", struct.pack("<f", x))[0]
        )
        run_until_complete(state)
        out_swish = struct.unpack(
            "<f", struct.pack("<I", state.regfile.get_r_acc_word(0))
        )[0]

        st2 = _make_state(
            """\
ACTIVATE lr0 silu;;
BKPT;;
"""
        )
        st2.regfile.set_cr(15, DType.E4)
        st2.regfile.set_lr(0, 1)
        st2.regfile.set_r_acc_word(
            0, struct.unpack("<I", struct.pack("<f", x))[0]
        )
        run_until_complete(st2)
        out_silu = struct.unpack(
            "<f", struct.pack("<I", st2.regfile.get_r_acc_word(0))
        )[0]
        assert abs(out_swish - out_silu) < 1e-7

    def test_activate_leaky_relu_respects_ipu_state_alpha(self):
        """``IpuState`` α overrides module defaults for ``ACTIVATE`` (not CR)."""
        x = -1.0
        alpha = 0.5
        state = _make_state(
            """\
ACTIVATE lr0 leaky_relu;;
BKPT;;
""",
            leaky_relu_alpha=alpha,
        )
        state.regfile.set_cr(15, DType.E4)
        state.regfile.set_lr(0, 1)
        state.regfile.set_r_acc_word(
            0, struct.unpack("<I", struct.pack("<f", x))[0]
        )
        run_until_complete(state)
        out = struct.unpack(
            "<f", struct.pack("<I", state.regfile.get_r_acc_word(0))
        )[0]
        exp = apply_activation(3, x, leaky_relu_alpha=alpha)
        assert abs(out - exp) < 1e-5

    def test_activate_leaky_relu_after_set_activation_alphas(self):
        x = -2.0
        alpha = 0.125
        state = _make_state(
            """\
ACTIVATE lr0 leaky_relu;;
BKPT;;
"""
        )
        state.set_activation_alphas(leaky_relu_alpha=alpha)
        state.regfile.set_cr(15, DType.E4)
        state.regfile.set_lr(0, 1)
        state.regfile.set_r_acc_word(
            0, struct.unpack("<I", struct.pack("<f", x))[0]
        )
        run_until_complete(state)
        out = struct.unpack(
            "<f", struct.pack("<I", state.regfile.get_r_acc_word(0))
        )[0]
        exp = apply_activation(3, x, leaky_relu_alpha=alpha)
        assert abs(out - exp) < 1e-5

    def test_activate_valid_elements_from_cr(self):
        state = _make_state(
            """\
ACTIVATE cr3 relu;;
BKPT;;
"""
        )
        state.regfile.set_cr(15, DType.INT8)
        state.regfile.set_cr(3, 1)
        state.regfile.set_r_acc_word(
            0, struct.unpack("<I", struct.pack("<i", -8))[0]
        )
        tail = struct.unpack("<I", struct.pack("<i", 123))[0]
        state.regfile.set_r_acc_word(1, tail)
        run_until_complete(state)
        assert (
            struct.unpack("<i", struct.pack("<I", state.regfile.get_r_acc_word(0)))[0]
            == 0
        )
        assert state.regfile.get_r_acc_word(1) == tail
