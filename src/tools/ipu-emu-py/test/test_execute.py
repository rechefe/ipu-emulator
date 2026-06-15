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
from ipu_emu.ipu_config import LR_CR_SCALAR_VALUE_MASK

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
    elu_alpha: float | None = None,
) -> IpuState:
    """Assemble *asm_code* and return a ready-to-run IpuState.

    Optional *cr* presets configuration registers before the program is loaded
    (e.g. constants read by ``SET lr* cr*``). Optional α kwargs are forwarded to
    :class:`IpuState` (emulator-only; not CR).
    """
    encoded = assemble(asm_code)
    decoded = [decode_instruction_word(w) for w in encoded]
    state = IpuState(elu_alpha=elu_alpha)
    if cr:
        for idx, val in cr.items():
            state.regfile.set_cr(idx, val)
    load_program(state, decoded)
    return state


def _run(
    asm_code: str,
    *,
    cr: dict[int, int] | None = None,
    elu_alpha: float | None = None,
    max_cycles: int = 100_000,
) -> IpuState:
    """Assemble, load, run, and return the final state."""
    state = _make_state(
        asm_code,
        cr=cr,
        elu_alpha=elu_alpha,
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
INC lr11 5;;
INC lr11 3;;
BKPT;;
""",
            cr={8: 10})
        assert state.regfile.get_lr(11) == 18  # 10 + 5 + 3

    def test_direct_access(self):
        state = IpuState()
        state.regfile.set_lr(11, 0xBEEF)
        state.regfile.set_lr(5, 0x12345)
        assert state.regfile.get_lr(11) == 0xBEEF
        assert state.regfile.get_lr(5) == 0x12345
        state.regfile.set_cr(2, 0xABCDE)
        assert state.regfile.get_cr(2) == 0xABCDE

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
SET lr2 cr9;;
ADD lr4 lr1 lr2;;
BKPT;;
""",
            cr={8: 200, 9: 11})
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
        assert state.regfile.get_lr(5) == (45 - 200) & LR_CR_SCALAR_VALUE_MASK

    def test_sub_lr_lr_imm5(self):
        state = _run("""\
SET lr2 cr8;;
SET lr4 cr9;;
SUB lr3 lr2 lr4;;
BKPT;;
""",
            cr={8: 100, 9: 30})
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


class TestIncDec:
    def test_inc_dest(self):
        state = _run("""\
SET lr5 cr8;;
INC lr5 7;;
BKPT;;
""",
            cr={8: 100})
        assert state.regfile.get_lr(5) == 107

    def test_dec_dest(self):
        state = _run("""\
SET lr5 cr8;;
DEC lr5 30;;
BKPT;;
""",
            cr={8: 100})
        assert state.regfile.get_lr(5) == 70

    def test_inc_dec_same_cycle(self):
        state = _run("""\
SET lr0 cr8;;
SET lr1 cr9;;
INC lr0 5;;
DEC lr1 2;;
BKPT;;
""",
            cr={8: 10, 9: 20})
        assert state.regfile.get_lr(0) == 15
        assert state.regfile.get_lr(1) == 18

    def test_decode_inc_imm_operand_field(self):
        """``INC`` immediate uses the union-derived LrIncDecImmediate field."""
        encoded = assemble("INC lr2 7;; BKPT;;")
        d = decode_instruction_word(encoded[0])
        assert d["lr_inst_0_token_0_lr_inst_opcode"] == 4  # inc
        assert d["lr_inst_0_token_2_lr_reg_field"] == 2
        assert d["lr_inst_0_token_1_lcr_reg_field"] == 7  # imm in shared lcr field


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
        assert d["lr_inst_0_token_1_lcr_reg_field"] == 8  # src = cr8
        assert d["lr_inst_0_token_3_lr_reg_field"] == 0   # unused field (default lr0)
        assert d["lr_inst_1_token_2_lr_reg_field"] == 5   # reg = lr5
        assert d["lr_inst_1_token_1_lcr_reg_field"] == 9   # src = cr9
        assert d["lr_inst_1_token_3_lr_reg_field"] == 0   # unused field (default lr0)
        assert d["lr_inst_2_token_2_lr_reg_field"] == 6   # reg = lr6
        assert d["lr_inst_2_token_1_lcr_reg_field"] == 10  # src = cr10
        assert d["lr_inst_2_token_3_lr_reg_field"] == 0   # unused field (default lr0)

    def test_decode_add_lcr_operand_field(self):
        """``ADD`` third operand uses LcrRegField (register-only)."""
        encoded = assemble("ADD lr2 lr1 cr7;; BKPT;;")
        d = decode_instruction_word(encoded[0])
        assert d["lr_inst_0_token_0_lr_inst_opcode"] == 1  # add
        assert d["lr_inst_0_token_2_lr_reg_field"] == 2   # dest = lr2
        assert d["lr_inst_0_token_3_lr_reg_field"] == 1   # src_a = lr1
        assert d["lr_inst_0_token_1_lcr_reg_field"] == 16 + 7  # src_b = cr7


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
        """INT8: r1=all-2, cyclic=all-3, MULT.RC.VV → acc should be 6 per word."""
        r1_data = bytes([2] * 128)
        cyclic_data = bytes([3] * 512)

        state = _make_state("""\
SET lr13 cr8;;
LDR_MULT_REG r1 lr13 cr0;;
SET lr14 cr9;;
SET lr15 cr10;;
LDR_CYCLIC_MULT_REG lr14 cr0 lr15;;
MULT.RC.VV lr0 r1 0 lr0;
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

    def test_cyclic_register_load_invalid_index_raises(self):
        """index must be one of the four R_CYCLIC slot boundaries — no implicit wrap."""
        from ipu_emu.ipu import EmulatorError

        state = _make_state("""\
SET lr0 cr8;;
SET lr1 cr9;;
LDR_CYCLIC_MULT_REG lr0 cr0 lr1;;
BKPT;;
""",
            cr={8: 20480, 9: 64})
        with pytest.raises(EmulatorError, match="index must be one of"):
            run_until_complete(state)

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

    def test_concurrent_load_and_store_same_cycle(self):
        """Load and store in one VLIW bundle: load resolves before store."""
        addr = 0x1000
        old_data = bytes([i & 0xFF for i in range(128)])
        state = _make_state("""\
SET lr0 cr8;;
LDR_MULT_REG r0 lr0 cr0;;
STR_ACC_REG lr0 cr0;;
BKPT;;
""",
            cr={8: addr})
        state.xmem.write_address(addr, old_data)
        state.regfile.set_r_acc_bytes(bytes([0xAB] * 512))
        run_until_complete(state)

        # Load saw pre-store memory; store overwrote memory afterward.
        assert state.regfile.get_r(0) == bytearray(old_data)
        stored = state.xmem.read_address(addr, 512)
        assert stored == state.regfile.get_r_acc_bytes()

    def test_mask_affects_multiplication(self):
        """First 64 bits of mask set → first 64 mult_res words active, rest zeroed."""
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
SET lr5 cr10;;
SET lr6 cr10;;
MULT.RC.VV lr6 r0 0 lr5;
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
            assert words[i] == 6, f"word {i} should be 6"
        for i in range(64, 128):
            assert words[i] == 0, f"word {i} should be masked to 0"

    def test_mask_with_shift(self):
        """Mask slot 1, shift_idx=+1 → 32 bits in slot 1 shift left by 1 (bits 1–32 masked out)."""
        r0_data = bytes([5] * 128)
        cyclic_data = bytes([4] * 512)
        # Mask slot 1 is at bytes 16..31; set first 4 bytes = 32 bits (lanes 0–31 of slot 1)
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
SET lr5 cr12;;
SET lr6 cr10;;
MULT.RC.VV lr6 r0 1 lr5;
ACC;;
SET lr9 cr13;;
STR_ACC_REG lr9 cr0;;
BKPT;;
""",
            cr={8: 4096, 9: 8192, 10: 0, 11: 12288, 12: 1, 13: 16384})
        state.xmem.write_address(0x1000, r0_data)
        state.xmem.write_address(0x2000, cyclic_data)
        state.xmem.write_address(0x3000, mask_data)
        run_until_complete(state)

        acc_bytes = state.xmem.read_address(0x4000, 512)
        words = struct.unpack_from("<128i", acc_bytes)
        assert words[0] == 0, "word 0 should be 0 (deactivated)"
        for i in range(1, 33):
            assert words[i] == 20, f"word {i} should be 20 (active)"
        for i in range(33, 128):
            assert words[i] == 0, f"word {i} should be 0 (deactivated)"


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

    def test_bge(self):
        state = _run("""\
SET lr0 cr8;;
SET lr1 cr9;;
BGE lr0 lr1 ge_branch;;
SET lr2 cr10;;
BKPT;;
ge_branch:
SET lr2 cr11;;
BKPT;;
""",
            cr={8: 6, 9: 5, 10: 0, 11: 1})
        assert state.regfile.get_lr(2) == 1

    def test_bge_not_taken(self):
        state = _run("""\
SET lr0 cr8;;
SET lr1 cr9;;
BGE lr0 lr1 ge_branch_not_taken;;
SET lr2 cr10;;
BKPT;;
ge_branch_not_taken:
SET lr2 cr11;;
BKPT;;
""",
            cr={8: 5, 9: 6, 10: 1, 11: 0})
        assert state.regfile.get_lr(2) == 1

    def test_bnz(self):
        state = _run("""\
SET lr0 cr8;;
BNZ lr0 nonzero_branch;;
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
BZ lr0 zero_branch;;
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

    def test_blt_negative_counter(self):
        """BLT sign-extends at 20 bits: -1 < 0 must branch (issue #142)."""
        state = _run("""\
SET lr0 cr8;;
SET lr1 cr9;;
BLT lr0 lr1 neg_branch;;
SET lr2 cr10;;
BKPT;;
neg_branch:
SET lr2 cr11;;
BKPT;;
""",
            cr={8: -1, 9: 0, 10: 0, 11: 1})
        assert state.regfile.get_lr(2) == 1

    def test_bge_negative_not_taken(self):
        """BGE sign-extends at 20 bits: -1 >= 0 must not branch (issue #142)."""
        state = _run("""\
SET lr0 cr8;;
SET lr1 cr9;;
BGE lr0 lr1 neg_ge_branch;;
SET lr2 cr10;;
BKPT;;
neg_ge_branch:
SET lr2 cr11;;
BKPT;;
""",
            cr={8: -1, 9: 0, 10: 1, 11: 0})
        assert state.regfile.get_lr(2) == 1

    def test_loop_with_cr_limit(self):
        """Loop using bne against a CR constant instead of an LR."""
        state = _make_state("""\
SET lr0 cr8;;
cr_loop_start:
INC lr0 1;;
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
INC lr0 1;;
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
BR cr2;;
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
        state.regfile.set_cr(2, target_addr)
        run_until_complete(state)
        assert state.regfile.get_lr(0) == 1


# ============================================================================
# Accumulator
# ============================================================================


class TestAccumulator:
    def test_acc_first(self):
        """ACC.FIRST sets r_acc to mult_res without adding previous r_acc."""
        state = _make_state(
            """\
ACC.FIRST;;
BKPT;;
"""
        )
        state.dtype = DType.INT8
        # Pre-fill acc with garbage; ACC.FIRST should ignore it
        for i in range(128):
            state.regfile.set_r_acc_word(i, 9999)
        mult_buf = state.regfile.raw("mult_res")
        for i in range(128):
            struct.pack_into("<i", mult_buf, i * 4, 7)
        run_until_complete(state)
        for i in range(128):
            assert state.regfile.get_r_acc_word(i) == 7, f"word {i}: expected 7, got {state.regfile.get_r_acc_word(i)}"

    def test_acc_stride_no_stride(self):
        """ACC.STRIDE with both strides off copies all 128 mult_res words to r_acc from start 0."""
        state = _make_state("""\
SET lr0 cr8;;
ACC.STRIDE 16 off off lr0;;
BKPT;;
""",
            cr={8: 0})
        state.dtype = DType.INT8
        mult_buf = state.regfile.raw("mult_res")
        for i in range(128):
            struct.pack_into("<i", mult_buf, i * 4, i)
        run_until_complete(state)
        for i in range(128):
            w = state.regfile.get_r_acc_word(i)
            assert w == i, f"word {i}: expected {i}, got {w}"

    def test_acc_stride_horizontal(self):
        """ACC.STRIDE with horizontal on: take every 2nd column → 64 elements at r_acc[0:64]."""
        state = _make_state("""\
SET lr0 cr8;;
ACC.STRIDE 16 on off lr0;;
BKPT;;
""",
            cr={8: 0})
        state.dtype = DType.INT8
        mult_buf = state.regfile.raw("mult_res")
        for i in range(128):
            struct.pack_into("<i", mult_buf, i * 4, i)
        run_until_complete(state)
        # Rows of 16: even columns 0,2,4,...,14 → 8 per row × 8 rows = 64 elements
        for out_i in range(64):
            row = out_i // 8
            col = (out_i % 8) * 2
            expected = row * 16 + col
            w = state.regfile.get_r_acc_word(out_i)
            assert w == expected, f"out[{out_i}]: expected {expected}, got {w}"

    def test_acc_stride_offset(self):
        """ACC.STRIDE with offset: (lr0 % 4)*32 is start index; 64 elements written at r_acc[32:96]."""
        state = _make_state("""\
SET lr0 cr8;;
ACC.STRIDE 16 on off lr0;;
BKPT;;
""",
            cr={8: 1})
        # lr0=1 → offset % 4 = 1 → start index 32. Horizontal on → 64 elements.
        state.dtype = DType.INT8
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
            row = out_i // 8
            col = (out_i % 8) * 2
            expected_src = row * 16 + col
            w = state.regfile.get_r_acc_word(32 + out_i)
            assert w == 100 + expected_src, f"word {32 + out_i}: expected {100 + expected_src}, got {w}"
        for i in range(96, 128):
            w = state.regfile.get_r_acc_word(i)
            assert w == 0, f"word {i} (after segment): expected 0, got {w}"

    def test_agg_sum_first_basic(self):
        """AGG.SUM.FIRST: sum all 128 MULT_RES lanes and write to R_ACC[dest] (clean init)."""
        state = _make_state(
            """\
AGG.SUM.FIRST LR0 1;;
BKPT;;
"""
        )
        state.dtype = DType.INT8
        state.set_cr_dstructure(128)
        state.regfile.set_lr(0, 127)  # dest = R_ACC[127]
        for i in range(128):
            state.regfile.set_mult_res_word(i, struct.unpack("<I", struct.pack("<i", 1))[0])
        run_until_complete(state)
        raw = state.regfile.get_r_acc_word(127)
        result = struct.unpack("<i", struct.pack("<I", raw))[0]
        assert result == 128, f"expected sum=128, got {result}"

    def test_agg_sum_first_ignores_existing_dest(self):
        """AGG.SUM.FIRST: existing R_ACC[dest] is NOT added to the result (clean initialisation).

        Dest is placed outside the active lane range so its value would be added
        as a seed in the non-FIRST variant, but must be ignored here.
        """
        state = _make_state(
            """\
AGG.SUM.FIRST LR0 0;;
BKPT;;
"""
        )
        state.dtype = DType.INT8
        state.set_cr_dstructure(4)  # only lanes 0..3 active
        state.regfile.set_lr(0, 127)  # dest = R_ACC[127], outside active range
        for i in range(128):
            state.regfile.set_mult_res_word(i, struct.unpack("<I", struct.pack("<i", 1))[0])
        state.regfile.set_r_acc_word(127, struct.unpack("<I", struct.pack("<i", 9999))[0])
        run_until_complete(state)
        raw = state.regfile.get_r_acc_word(127)
        result = struct.unpack("<i", struct.pack("<I", raw))[0]
        # sum of active MULT_RES lanes 0..3 = 4; existing dest (9999) is NOT added
        assert result == 4, f"expected sum=4 (not 4+9999), got {result}"

    def test_agg_sum_first_uses_valid_elements(self):
        """AGG.SUM.FIRST: only active MULT_RES prefix contributes; tail is excluded."""
        state = _make_state(
            """\
AGG.SUM.FIRST LR0 0;;
BKPT;;
"""
        )
        state.dtype = DType.INT8
        state.set_cr_dstructure(4)
        state.regfile.set_lr(0, 127)
        for i in range(128):
            v = 10 if i < 4 else 1000
            state.regfile.set_mult_res_word(i, struct.unpack("<I", struct.pack("<i", v))[0])
        run_until_complete(state)
        raw = state.regfile.get_r_acc_word(127)
        result = struct.unpack("<i", struct.pack("<I", raw))[0]
        assert result == 40, f"expected sum of 4 tens = 40, got {result}"

    def test_agg_sum_first_full_xmem_row_overrides_valid_elements(self):
        """AGG.SUM.FIRST with full_xmem_row=1 uses all 128 MULT_RES lanes regardless of CR15."""
        state = _make_state(
            """\
AGG.SUM.FIRST LR0 1;;
BKPT;;
"""
        )
        state.dtype = DType.INT8
        state.set_cr_dstructure(valid_elements=4)
        state.regfile.set_lr(0, 127)
        for i in range(128):
            state.regfile.set_mult_res_word(i, struct.unpack("<I", struct.pack("<i", 1))[0])
        run_until_complete(state)
        raw = state.regfile.get_r_acc_word(127)
        result = struct.unpack("<i", struct.pack("<I", raw))[0]
        assert result == 128, f"expected sum of all 128 lanes = 128, got {result}"

    def test_agg_sum_accumulates(self):
        """AGG.SUM: sum of active MULT_RES lanes is ADDED to existing R_ACC[dest] (running acc).

        Dest is placed outside the active lane range so it acts as a pure accumulator
        that is not double-counted in the sum.
        """
        state = _make_state(
            """\
AGG.SUM LR0 0;;
BKPT;;
"""
        )
        state.dtype = DType.INT8
        state.set_cr_dstructure(64)  # only lanes 0..63 active
        state.regfile.set_lr(0, 127)  # dest = R_ACC[127], outside active range
        for i in range(128):
            state.regfile.set_mult_res_word(i, struct.unpack("<I", struct.pack("<i", 1))[0])
        state.regfile.set_r_acc_word(127, struct.unpack("<I", struct.pack("<i", 50))[0])
        run_until_complete(state)
        # sum(MULT_RES[0..63]) = 64; R_ACC[127] was 50 → result = 64 + 50 = 114
        raw = state.regfile.get_r_acc_word(127)
        result = struct.unpack("<i", struct.pack("<I", raw))[0]
        assert result == 114, f"expected 64+50=114, got {result}"

    def test_agg_max_first_basic(self):
        """AGG.MAX.FIRST: max of all 128 MULT_RES lanes, no seed from dest."""
        state = _make_state(
            """\
AGG.MAX.FIRST LR0 1;;
BKPT;;
"""
        )
        state.dtype = DType.INT8
        state.set_cr_dstructure(128)
        state.regfile.set_lr(0, 127)
        for i in range(128):
            state.regfile.set_mult_res_word(i, struct.unpack("<I", struct.pack("<i", 5 + (i % 10)))[0])
        state.regfile.set_mult_res_word(127, struct.unpack("<I", struct.pack("<i", 9999))[0])
        run_until_complete(state)
        raw = state.regfile.get_r_acc_word(127)
        result = struct.unpack("<i", struct.pack("<I", raw))[0]
        assert result == 9999, f"expected max=9999 (from MULT_RES slot 127), got {result}"

    def test_agg_max_first_ignores_garbage_dest(self):
        """AGG.MAX.FIRST: existing R_ACC[dest] is NOT used as a seed.

        Dest is placed outside the active lane range and pre-loaded with a
        large garbage value; a seeded implementation would return it, the
        clean-init implementation must return the max of the active MULT_RES lanes.
        """
        state = _make_state(
            """\
AGG.MAX.FIRST LR0 0;;
BKPT;;
"""
        )
        state.dtype = DType.INT8
        state.set_cr_dstructure(64)  # only lanes 0..63 active
        state.regfile.set_lr(0, 127)  # dest = R_ACC[127], outside active range
        for i in range(128):
            state.regfile.set_mult_res_word(i, struct.unpack("<I", struct.pack("<i", 5))[0])
        state.regfile.set_r_acc_word(127, struct.unpack("<I", struct.pack("<i", 9999))[0])
        run_until_complete(state)
        raw = state.regfile.get_r_acc_word(127)
        result = struct.unpack("<i", struct.pack("<I", raw))[0]
        assert result == 5, f"expected max of active MULT_RES lanes = 5 (not R_ACC seed 9999), got {result}"

    def test_agg_max_first_zero_active_writes_identity_seed(self):
        """AGG.MAX.FIRST with valid_elements=0: dest gets the identity seed (INT32_MIN)."""
        state = _make_state(
            """\
AGG.MAX.FIRST LR0 0;;
BKPT;;
"""
        )
        state.dtype = DType.INT8
        state.set_cr_dstructure(0)  # no active lanes
        state.regfile.set_lr(0, 7)
        state.regfile.set_r_acc_word(7, struct.unpack("<I", struct.pack("<i", 1234))[0])
        run_until_complete(state)
        raw = state.regfile.get_r_acc_word(7)
        result = struct.unpack("<i", struct.pack("<I", raw))[0]
        assert result == -2147483648, f"expected INT32_MIN identity seed, got {result}"

    def test_agg_max_first_masks_tail(self):
        """AGG.MAX.FIRST: MULT_RES tail lanes beyond valid_elements are excluded."""
        state = _make_state(
            """\
AGG.MAX.FIRST LR0 0;;
BKPT;;
"""
        )
        state.dtype = DType.INT8
        state.set_cr_dstructure(50)
        state.regfile.set_lr(0, 127)
        for i in range(128):
            v = 3 if i < 50 else 9999
            state.regfile.set_mult_res_word(i, struct.unpack("<I", struct.pack("<i", v))[0])
        run_until_complete(state)
        raw = state.regfile.get_r_acc_word(127)
        result = struct.unpack("<i", struct.pack("<I", raw))[0]
        assert result == 3, f"expected max of active MULT_RES prefix = 3, got {result}"

    def test_agg_max_seed_wins(self):
        """AGG.MAX: existing R_ACC[dest] seed beats all active MULT_RES lanes — dest unchanged."""
        state = _make_state(
            """\
AGG.MAX LR0 1;;
BKPT;;
"""
        )
        state.dtype = DType.INT8
        state.set_cr_dstructure(128)
        state.regfile.set_lr(0, 127)
        for i in range(128):
            state.regfile.set_mult_res_word(i, struct.unpack("<I", struct.pack("<i", 10))[0])
        state.regfile.set_r_acc_word(127, struct.unpack("<I", struct.pack("<i", 99))[0])
        run_until_complete(state)
        raw = state.regfile.get_r_acc_word(127)
        result = struct.unpack("<i", struct.pack("<I", raw))[0]
        assert result == 99, f"expected seed 99 to remain (beats all MULT_RES lanes=10), got {result}"

    def test_agg_max_lane_wins(self):
        """AGG.MAX: an active MULT_RES lane beats the existing R_ACC[dest] seed — dest updated."""
        state = _make_state(
            """\
AGG.MAX LR0 0;;
BKPT;;
"""
        )
        state.dtype = DType.INT8
        state.set_cr_dstructure(100)
        state.regfile.set_lr(0, 127)
        for i in range(128):
            v = 5 if i < 100 else 1
            state.regfile.set_mult_res_word(i, struct.unpack("<I", struct.pack("<i", v))[0])
        state.regfile.set_mult_res_word(63, struct.unpack("<I", struct.pack("<i", 77))[0])
        state.regfile.set_r_acc_word(127, struct.unpack("<I", struct.pack("<i", 3))[0])
        run_until_complete(state)
        raw = state.regfile.get_r_acc_word(127)
        result = struct.unpack("<i", struct.pack("<I", raw))[0]
        assert result == 77, f"expected max=77, got {result}"

    def test_agg_and_activate_same_cycle_use_snapshot(self):
        """AGG.SUM.FIRST in ACC slot and ACTIVATE in AAQ slot issued together.

        AGG reads from MULT_RES (live) and writes r_acc[dest].
        ACTIVATE reads from the cycle-start snapshot of r_acc (not the AGG write).
        """
        state = _make_state(
            """\
AGG.SUM.FIRST LR0 1; ACTIVATE relu 1;;
BKPT;;
"""
        )
        state.dtype = DType.INT8
        state.regfile.set_lr(0, 0)  # AGG dest = r_acc[0]
        import struct as _struct
        for i in range(128):
            # MULT_RES lanes = 3 (source for AGG.SUM.FIRST)
            state.regfile.set_mult_res_word(i, _struct.unpack("<I", _struct.pack("<i", 3))[0])
            # r_acc lanes = 3 (source for ACTIVATE snapshot)
            state.regfile.set_r_acc_word(i, _struct.unpack("<I", _struct.pack("<i", 3))[0])

        run_until_complete(state)

        # AGG.SUM.FIRST must have summed MULT_RES (128 lanes × 3 = 384) into r_acc[0]
        raw_agg = state.regfile.get_r_acc_word(0)
        agg_result = _struct.unpack("<i", _struct.pack("<I", raw_agg))[0]
        assert agg_result == 384, f"AGG saw wrong mult_res: expected 384, got {agg_result}"

        # ACTIVATE relu must have applied relu to the snapshot r_acc (all lanes 3 → 3)
        # NOT the post-AGG r_acc where lane 0 = 384.
        # In INT8 mode ACTIVATE writes int32 to post_aaq_reg.
        post = state.regfile.raw("post_aaq_reg")
        lane0_post = _struct.unpack_from("<i", post, 0)[0]
        assert lane0_post == 3, (
            f"ACTIVATE saw wrong r_acc lane 0: expected 3, got {lane0_post}"
        )


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
        assert d["lr_inst_0_token_1_lcr_reg_field"] == 8  # src = cr8


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
SET lr5 cr10;;
SET lr6 cr10;;
MULT.RC.VV lr6 r0 0 lr5;
ACC;;
BKPT;;
""",
            cr={8: 4096, 9: 8192, 10: 0})
        # Set dtype to E4 (fp8_e4)
        state.dtype = DType.E4
        state.xmem.write_address(0x1000, r0_data)
        state.xmem.write_address(0x2000, cyclic_data)
        run_until_complete(state)

        # Read back acc as floats
        acc_raw = state.regfile.raw("r_acc")
        for i in range(128):
            val = struct.unpack_from("<f", acc_raw, i * 4)[0]
            assert abs(val - 2.0) < 0.01, f"acc word {i}: expected 2.0, got {val}"


# ============================================================================
# MULT.RC.VE
# ============================================================================


class TestMultRcVe:
    """Tests for the MULT.RC.VE instruction."""

    # ------------------------------------------------------------------
    # MULT.RC.VE with a CR-encoded scalar source
    # ------------------------------------------------------------------

    def test_mult_rc_ve_cr_int8(self):
        """MULT.RC.VE INT8: scalar from CR × RC elements."""
        # CR scalar byte = 3, RC elements = all 2 → each result = 3*2 = 6
        cyclic_data = bytes([2] * 512)

        state = _make_state("""\
SET lr0 cr8;;
SET lr1 cr9;;
LDR_CYCLIC_MULT_REG lr0 cr0 lr1;;
SET lr2 cr9;;
SET lr4 cr9;;
MULT.RC.VE lr2 cr3 0 lr4;
ACC;;
BKPT;;
""",
            cr={8: 4096, 9: 0})
        state.dtype = DType.INT8
        state.regfile.set_cr(3, 3)  # scalar = low byte = 3
        state.xmem.write_address(0x1000, cyclic_data)
        run_until_complete(state)

        acc_raw = state.regfile.raw("r_acc")
        for i in range(128):
            val = struct.unpack_from("<i", acc_raw, i * 4)[0]
            assert val == 6, f"acc word {i}: expected 6, got {val}"

    def test_mult_rc_ve_cr_negative_int8(self):
        """MULT.RC.VE INT8: signed negative scalar × positive RC elements."""
        # CR scalar byte = 0xFE = -2 (signed int8), RC elements = 5 → result = -10
        cyclic_data = bytes([5] * 512)

        state = _make_state("""\
SET lr0 cr8;;
SET lr1 cr9;;
LDR_CYCLIC_MULT_REG lr0 cr0 lr1;;
SET lr2 cr9;;
SET lr4 cr9;;
MULT.RC.VE lr2 cr2 0 lr4;
ACC;;
BKPT;;
""",
            cr={8: 4096, 9: 0})
        state.dtype = DType.INT8
        state.regfile.set_cr(2, 0xFE)  # low byte 0xFE = int8(-2)
        state.xmem.write_address(0x1000, cyclic_data)
        run_until_complete(state)

        acc_raw = state.regfile.raw("r_acc")
        for i in range(128):
            val = struct.unpack_from("<i", acc_raw, i * 4)[0]
            assert val == -10, f"acc word {i}: expected -10, got {val}"

    def test_mult_rc_ve_cr_wraps_at_rc_boundary(self):
        """MULT.RC.VE: RC indices wrap modulo 512 (cyclic, no padding)."""
        # rc_idx = 450, so bytes 450..511 then 0..65 of RC are used — all rc_fill.
        rc_fill = 4
        scalar = 7

        state = _make_state("""\
SET lr2 cr8;;
SET lr4 cr9;;
MULT.RC.VE lr2 cr3 0 lr4;
ACC;;
BKPT;;
""",
            cr={8: 450, 9: 0})
        state.dtype = DType.INT8
        state.regfile.set_cr(3, scalar)
        # Fill the full 512-byte cyclic register directly
        state.regfile.set_r_cyclic_at(0, bytes([rc_fill] * 512))
        run_until_complete(state)

        acc_raw = state.regfile.raw("r_acc")
        for i in range(128):
            val = struct.unpack_from("<i", acc_raw, i * 4)[0]
            assert val == scalar * rc_fill, f"word {i}: expected {scalar * rc_fill}, got {val}"

    def test_mult_rc_ve_cr_fp8e4m3(self):
        """MULT.RC.VE fp8_e4: scalar 1.0 × RC elements 1.0 → result 1.0 each."""
        from ipu_emu.ipu_math import _float32_to_fp8_scalar

        one_fp8 = _float32_to_fp8_scalar(1.0, 4)
        cyclic_data = bytes([one_fp8] * 512)

        state = _make_state("""\
SET lr0 cr8;;
SET lr1 cr9;;
LDR_CYCLIC_MULT_REG lr0 cr0 lr1;;
SET lr2 cr9;;
SET lr4 cr9;;
MULT.RC.VE lr2 cr5 0 lr4;
ACC;;
BKPT;;
""",
            cr={8: 4096, 9: 0})
        state.dtype = DType.E4
        state.regfile.set_cr(5, one_fp8)  # scalar = 1.0 in fp8_e4
        state.xmem.write_address(0x1000, cyclic_data)
        run_until_complete(state)

        acc_raw = state.regfile.raw("r_acc")
        for i in range(128):
            val = struct.unpack_from("<f", acc_raw, i * 4)[0]
            assert abs(val - 1.0) < 0.01, f"acc word {i}: expected 1.0, got {val}"

    def test_mult_rc_ve_cr_wraps_fp8e5m2(self):
        """MULT.RC.VE fp8_e5: RC indices wrap modulo 512 (cyclic, no padding)."""
        from ipu_emu.ipu_math import _float32_to_fp8_scalar

        two_fp8 = _float32_to_fp8_scalar(2.0, 5)
        scalar_fp8 = _float32_to_fp8_scalar(3.0, 5)

        state = _make_state("""\
SET lr2 cr8;;
SET lr4 cr9;;
MULT.RC.VE lr2 cr6 0 lr4;
ACC;;
BKPT;;
""",
            cr={8: 500, 9: 0})
        state.dtype = DType.E5
        state.regfile.set_cr(6, scalar_fp8)  # scalar = 3.0
        # Fill the full 512-byte cyclic register directly with 2.0 in fp8_e5
        state.regfile.set_r_cyclic_at(0, bytes([two_fp8] * 512))
        run_until_complete(state)

        acc_raw = state.regfile.raw("r_acc")
        for i in range(128):  # wraps modulo 512: every word reads 2.0 → 3.0 * 2.0 = 6.0
            val = struct.unpack_from("<f", acc_raw, i * 4)[0]
            assert abs(val - 6.0) < 0.1, f"word {i}: expected 6.0, got {val}"

    # ------------------------------------------------------------------
    # MULT.RC.VV / MULT.RC.VE behave correctly alongside each other
    # ------------------------------------------------------------------

    def test_mult_rc_vv_still_works(self):
        """MULT.RC.VV still works correctly after adding MULT.RC.VE."""
        r0_data = bytes([4] * 128)
        cyclic_data = bytes([5] * 512)

        state = _make_state("""\
SET lr0 cr8;;
LDR_MULT_REG r0 lr0 cr0;;
SET lr1 cr9;;
SET lr2 cr10;;
LDR_CYCLIC_MULT_REG lr1 cr0 lr2;;
MULT.RC.VV lr2 r0 0 lr2;
ACC;;
BKPT;;
""",
            cr={8: 4096, 9: 8192, 10: 0})
        state.dtype = DType.INT8
        state.xmem.write_address(0x1000, r0_data)
        state.xmem.write_address(0x2000, cyclic_data)
        run_until_complete(state)

        acc_raw = state.regfile.raw("r_acc")
        for i in range(128):
            val = struct.unpack_from("<i", acc_raw, i * 4)[0]
            assert val == 20, f"acc word {i}: expected 20, got {val}"

    def test_mult_rc_ve_lr_scalar(self):
        """MULT.RC.VE: src as LR uses the LR's value to index into R0 (combined Ra buffer)."""
        cyclic_data = bytes([6] * 512)
        r0_data = bytearray(128)
        r0_data[0] = 3  # LR value 0 → Ra[0] = 3

        state = _make_state("""\
SET lr0 cr8;;
LDR_MULT_REG r0 lr0 cr0;;
SET lr1 cr9;;
SET lr2 cr10;;
LDR_CYCLIC_MULT_REG lr1 cr0 lr2;;
MULT.RC.VE lr2 lr2 0 lr2;
ACC;;
BKPT;;
""",
            cr={8: 4096, 9: 8192, 10: 0})
        state.dtype = DType.INT8
        state.xmem.write_address(0x1000, bytes(r0_data))
        state.xmem.write_address(0x2000, cyclic_data)
        run_until_complete(state)

        acc_raw = state.regfile.raw("r_acc")
        for i in range(128):
            val = struct.unpack_from("<i", acc_raw, i * 4)[0]
            assert val == 18, f"acc word {i}: expected 18, got {val}"

    def test_mult_rc_ve_r1_scalar(self):
        """MULT.RC.VE: src LR value in [128, 255] addresses R1[value - 128] instead of R0."""
        r0_data = bytearray(128)  # all zeros — must not be picked
        r1_data = bytearray(128)
        r1_data[0] = 7  # LR value=128 → r1[0] = 7
        cyclic_data = bytes([4] * 512)

        state = _make_state("""\
SET lr0 cr8;;
LDR_MULT_REG r0 lr0 cr0;;
SET lr0 cr9;;
LDR_MULT_REG r1 lr0 cr0;;
SET lr1 cr10;;
SET lr2 cr11;;
LDR_CYCLIC_MULT_REG lr1 cr0 lr2;;
SET lr3 cr12;;
MULT.RC.VE lr2 lr3 0 lr2;
ACC;;
BKPT;;
""",
            cr={8: 4096, 9: 4352, 10: 8192, 11: 0, 12: 128})
        state.dtype = DType.INT8
        state.xmem.write_address(0x1000, bytes(r0_data))
        state.xmem.write_address(0x1100, bytes(r1_data))
        state.xmem.write_address(0x2000, cyclic_data)
        run_until_complete(state)

        acc_raw = state.regfile.raw("r_acc")
        for i in range(128):
            val = struct.unpack_from("<i", acc_raw, i * 4)[0]
            assert val == 28, f"acc word {i}: expected 28 (r1[0]=7 × cyclic[i]=4), got {val}"


class TestMultRcVs:
    """MULT.RC.VS — self-multiply (square) of RC vector elements."""

    def test_mult_rc_vs_squares_rc(self):
        """Each RC lane multiplied by itself (4 × 4 = 16)."""
        cyclic_data = bytes([4] * 512)

        state = _make_state("""\
SET lr0 cr8;;
SET lr1 cr9;;
LDR_CYCLIC_MULT_REG lr0 cr0 lr1;;
SET lr5 cr9;;
MULT.RC.VS lr1 0 lr5;
ACC;;
BKPT;;
""",
            cr={8: 4096, 9: 0})
        state.dtype = DType.INT8
        state.xmem.write_address(0x1000, cyclic_data)
        run_until_complete(state)

        acc_raw = state.regfile.raw("r_acc")
        for i in range(128):
            val = struct.unpack_from("<i", acc_raw, i * 4)[0]
            assert val == 16, f"acc word {i}: expected 16 (4×4), got {val}"

    def test_mult_rc_vs_wraps_at_boundary(self):
        """rc_idx near the 512-byte boundary wraps cyclically (3 × 3 = 9)."""
        cyclic_data = bytes([3] * 512)

        state = _make_state("""\
SET lr2 cr10;;
SET lr5 cr11;;
MULT.RC.VS lr2 0 lr5;
ACC;;
BKPT;;
""",
            cr={10: 450, 11: 0})
        state.dtype = DType.INT8
        state.regfile.set_r_cyclic_at(0, cyclic_data)
        run_until_complete(state)

        acc_raw = state.regfile.raw("r_acc")
        for i in range(128):
            val = struct.unpack_from("<i", acc_raw, i * 4)[0]
            assert val == 9, f"acc word {i}: expected 9 (3×3), got {val}"

    def test_mult_rc_vs_mask_zeroes_lanes(self):
        """mask_offset/mask_shift still gate lanes (first 64 active, rest zeroed)."""
        cyclic_data = bytes([3] * 512)   # squared → 9
        mask_data = bytearray(128)
        for i in range(8):           # 64 bits set → first 64 lanes active
            mask_data[i] = 0xFF

        state = _make_state("""\
SET lr0 cr8;;
SET lr1 cr9;;
LDR_CYCLIC_MULT_REG lr0 cr0 lr1;;
SET lr3 cr10;;
LDR_MULT_MASK_REG lr3 cr0;;
SET lr5 cr9;;
MULT.RC.VS lr1 0 lr5;
ACC;;
BKPT;;
""",
            cr={8: 4096, 9: 0, 10: 8192})
        state.dtype = DType.INT8
        state.xmem.write_address(0x1000, cyclic_data)
        state.xmem.write_address(0x2000, bytes(mask_data))
        run_until_complete(state)

        acc_raw = state.regfile.raw("r_acc")
        for i in range(64):
            val = struct.unpack_from("<i", acc_raw, i * 4)[0]
            assert val == 9, f"word {i}: expected 9 (3×3), got {val}"
        for i in range(64, 128):
            val = struct.unpack_from("<i", acc_raw, i * 4)[0]
            assert val == 0, f"word {i} should be masked to 0, got {val}"


# ============================================================================
# Sequential shift-and-AND mask generation (issue #99)
# ============================================================================


class TestMaskShiftSequential:
    """Sequential shift-and-AND mask generation for mult mask_shift operand.

    mask_shift_idx ∈ [−3, +3] selects from 7 pre-generated masks:
      idx 0  → base mask (unmodified)
      idx −k → shift right k times, AND with partition_vector after each step
      idx +k → shift left  k times, AND with partition_vector after each step

    The partition_vector is derived from CR15.partition (0 = no partitioning).
    """

    _RC_ADDR = 0x1000   # xmem address for R_CYCLIC data
    _MASK_ADDR = 0x2000  # xmem address for mask data
    _RC_CR = 8           # CR holding R_CYCLIC xmem address
    _MASK_CR = 9         # CR holding mask xmem address
    _SHIFT_CR = 2        # CR holding mask_shift_idx (set per test)

    _ASM = """\
SET lr0 cr8;;
SET lr1 cr0;;
LDR_CYCLIC_MULT_REG lr0 cr0 lr1;;
SET lr3 cr9;;
LDR_MULT_MASK_REG lr3 cr0;;
SET lr5 cr2;;
MULT.RC.VS lr1 0 lr5;
ACC;;
BKPT;;
"""

    def _run_mask_shift(
        self,
        base_mask_bits: int,
        *,
        shift_idx: int,
        partition: int = 0,
    ) -> bytearray:
        """Run MULT.RC.VS with given 128-bit base mask and mask_shift_idx; return r_acc."""
        mask_bytes = base_mask_bits.to_bytes(16, byteorder="little") + bytes(112)
        state = _make_state(
            self._ASM,
            cr={
                self._RC_CR: self._RC_ADDR,
                self._MASK_CR: self._MASK_ADDR,
                self._SHIFT_CR: shift_idx & 0xFFFFFFFF,
            },
        )
        state.dtype = DType.INT8
        state.set_cr_dstructure(valid_elements=128, partition=partition)
        state.xmem.write_address(self._RC_ADDR, bytes([2] * 512))
        state.xmem.write_address(self._MASK_ADDR, mask_bytes)
        run_until_complete(state)
        return state.regfile.raw("r_acc")

    def _suppressed(self, acc_raw: bytearray, lane: int) -> bool:
        return struct.unpack_from("<i", acc_raw, lane * 4)[0] == 0

    # ------------------------------------------------------------------
    # shift indices with partition=0 (no partitioning — unconstrained shift)
    # ------------------------------------------------------------------

    def test_idx_0_base_mask_unchanged(self):
        """idx=0: base mask used as-is; only lane 32 is active, all others suppressed."""
        acc = self._run_mask_shift(1 << 32, shift_idx=0, partition=0)
        assert not self._suppressed(acc, 32)
        for i in range(128):
            if i != 32:
                assert self._suppressed(acc, i), f"lane {i} should be suppressed"

    def test_idx_minus1_shifts_right_by_1(self):
        """idx=−1: base mask shifts right by 1; bit 32 → bit 31."""
        acc = self._run_mask_shift(1 << 32, shift_idx=-1, partition=0)
        assert not self._suppressed(acc, 31)
        for i in range(128):
            if i != 31:
                assert self._suppressed(acc, i), f"lane {i} should be suppressed"

    def test_idx_minus2_shifts_right_by_2(self):
        """idx=−2: base mask shifts right twice; bit 32 → bit 30."""
        acc = self._run_mask_shift(1 << 32, shift_idx=-2, partition=0)
        assert not self._suppressed(acc, 30)
        for i in range(128):
            if i != 30:
                assert self._suppressed(acc, i), f"lane {i} should be suppressed"

    def test_idx_minus3_shifts_right_by_3(self):
        """idx=−3: base mask shifts right three times; bit 32 → bit 29."""
        acc = self._run_mask_shift(1 << 32, shift_idx=-3, partition=0)
        assert not self._suppressed(acc, 29)
        for i in range(128):
            if i != 29:
                assert self._suppressed(acc, i), f"lane {i} should be suppressed"

    def test_idx_plus1_shifts_left_by_1(self):
        """idx=+1: base mask shifts left by 1; bit 32 → bit 33."""
        acc = self._run_mask_shift(1 << 32, shift_idx=+1, partition=0)
        assert not self._suppressed(acc, 33)
        for i in range(128):
            if i != 33:
                assert self._suppressed(acc, i), f"lane {i} should be suppressed"

    def test_idx_plus2_shifts_left_by_2(self):
        """idx=+2: base mask shifts left twice; bit 32 → bit 34."""
        acc = self._run_mask_shift(1 << 32, shift_idx=+2, partition=0)
        assert not self._suppressed(acc, 34)
        for i in range(128):
            if i != 34:
                assert self._suppressed(acc, i), f"lane {i} should be suppressed"

    def test_idx_plus3_shifts_left_by_3(self):
        """idx=+3: base mask shifts left three times; bit 32 → bit 35."""
        acc = self._run_mask_shift(1 << 32, shift_idx=+3, partition=0)
        assert not self._suppressed(acc, 35)
        for i in range(128):
            if i != 35:
                assert self._suppressed(acc, i), f"lane {i} should be suppressed"

    # ------------------------------------------------------------------
    # partition=2 (2 partitions of 64): verify boundary enforcement
    # ------------------------------------------------------------------

    def test_left_shift_cleared_at_partition_boundary(self):
        """A bit at the last position of partition 0 cannot shift into partition 1.

        With partition=2 (step=64), bit 63 is the last element of partition 0.
        Shifting left by 1 would move it to bit 64 (start of partition 1),
        but partition_vector[64]=0 clears it → mask = 0 → all lanes deactivated.
        """
        acc = self._run_mask_shift(1 << 63, shift_idx=+1, partition=2)
        for i in range(128):
            assert self._suppressed(acc, i), f"lane {i} should be suppressed"

    def test_right_shift_stays_within_partition(self):
        """A bit inside partition 1 shifts freely within partition 1.

        With partition=2, bit 65 shifted right by 1 → bit 64 (still inside partition 1).
        inverse_partition_vector[64]=1, so the bit is NOT cleared → lane 64 is active.
        """
        acc = self._run_mask_shift(1 << 65, shift_idx=-1, partition=2)
        assert not self._suppressed(acc, 64)
        for i in range(128):
            if i != 64:
                assert self._suppressed(acc, i), f"lane {i} should be suppressed"

    def test_shift_stays_within_partition(self):
        """A bit inside partition 0 shifts freely within partition 0.

        With partition=2, bit 33 shifted left by 1 → bit 34 (still in partition 0).
        """
        acc = self._run_mask_shift(1 << 33, shift_idx=+1, partition=2)
        assert not self._suppressed(acc, 34)
        for i in range(128):
            if i != 34:
                assert self._suppressed(acc, i), f"lane {i} should be suppressed"

    def test_partition_4_boundary_constraint(self):
        """partition=4 (4 partitions of 32): bit at last pos of partition cannot cross → all lanes deactivated."""
        acc = self._run_mask_shift(1 << 31, shift_idx=+1, partition=4)
        for i in range(128):
            assert self._suppressed(acc, i), f"lane {i} should be suppressed (partition 4)"

    def test_partition_8_boundary_constraint(self):
        """partition=8 (8 partitions of 16): bit at last pos of partition cannot cross → all lanes deactivated."""
        acc = self._run_mask_shift(1 << 15, shift_idx=+1, partition=8)
        for i in range(128):
            assert self._suppressed(acc, i), f"lane {i} should be suppressed (partition 8)"

    def test_sequential_and_enforced_on_each_step(self):
        """Sequential AND is applied after EACH shift step, not just at the end.

        With partition=2, shift_idx=−2, using inverse_partition_vector (0 at end of each group):
          step 1: (1<<65 >> 1) = 1<<64; inverse_pv[64]=1 → 1<<64 (not cleared)
          step 2: (1<<64 >> 1) = 1<<63; inverse_pv[63]=0 → 0 (cleared at end of group 0)
        Result: all lanes deactivated (bit cleared at second step → mask = 0).
        Without the per-step AND, a 2-bit right shift of bit 65 would land at 63
        and survive since inverse_pv[63] is never checked in a single-step AND.
        """
        acc = self._run_mask_shift(1 << 65, shift_idx=-2, partition=2)
        for i in range(128):
            assert self._suppressed(acc, i), f"lane {i} should be suppressed"


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
        state.regfile.set_post_aaq_reg(bytearray(buf))

    def test_aaq_basic_clamp(self):
        """Direct clamp: values already in [-128, 127] pass through unchanged."""
        state = IpuState()
        state.dtype = DType.INT8
        # Lanes 0..127 hold the signed value (i - 64): -64..63, all in range.
        self._set_acc_words(state, [i - 64 for i in range(128)])

        encoded = assemble("aaq 0;;\nBKPT;;")
        from ipu_emu.execute import decode_instruction_word
        from ipu_emu.emulator import load_program, run_until_complete
        decoded = [decode_instruction_word(w) for w in encoded]
        load_program(state, decoded)
        run_until_complete(state)

        result = state.regfile.get_post_aaq_reg()
        for i in range(128):
            expected = i - 64
            assert result[i] == (expected & 0xFF), f"byte {i}: expected {expected & 0xFF}, got {result[i]}"
        assert result[128:] == bytearray(384), "tail of POST_AAQ_REG should be cleared"

    def test_aaq_all_zeros(self):
        """All-zero accumulator quantizes to all-zero bytes."""
        state = IpuState()
        state.dtype = DType.INT8
        self._set_acc_words(state, [0] * 128)

        encoded = assemble("aaq 0;;\nBKPT;;")
        from ipu_emu.execute import decode_instruction_word
        from ipu_emu.emulator import load_program, run_until_complete
        decoded = [decode_instruction_word(w) for w in encoded]
        load_program(state, decoded)
        run_until_complete(state)

        assert state.regfile.get_post_aaq_reg() == bytearray(512)

    def test_aaq_positive_clamp(self):
        """Direct clamp: values above 127 saturate to 127."""
        state = IpuState()
        state.dtype = DType.INT8
        # 127 stays 127; large positive int32 saturates to 127.
        values = [127] * 64 + [0x7FFFFFFF] * 64
        self._set_acc_words(state, values)

        encoded = assemble("aaq 0;;\nBKPT;;")
        from ipu_emu.execute import decode_instruction_word
        from ipu_emu.emulator import load_program, run_until_complete
        decoded = [decode_instruction_word(w) for w in encoded]
        load_program(state, decoded)
        run_until_complete(state)

        result = state.regfile.get_post_aaq_reg()
        for i in range(128):
            assert result[i] == 127, f"byte {i}: expected 127, got {result[i]}"
        assert result[128:] == bytearray(384)

    def test_aaq_negative_values(self):
        """Direct clamp: in-range negatives pass through; below -128 saturates to -128."""
        state = IpuState()
        state.dtype = DType.INT8
        # -1 stays -1 (0xFF); large negative int32 saturates to -128 (0x80).
        values = [-1] * 64 + [-(1 << 30)] * 64
        self._set_acc_words(state, values)

        encoded = assemble("aaq 0;;\nBKPT;;")
        from ipu_emu.execute import decode_instruction_word
        from ipu_emu.emulator import load_program, run_until_complete
        decoded = [decode_instruction_word(w) for w in encoded]
        load_program(state, decoded)
        run_until_complete(state)

        result = state.regfile.get_post_aaq_reg()
        for i in range(64):
            assert result[i] == 0xFF, f"byte {i}: expected 0xFF (-1), got {result[i]}"
        for i in range(64, 128):
            assert result[i] == 0x80, f"byte {i}: expected 0x80 (-128), got {result[i]}"
        assert result[128:] == bytearray(384)

    def test_aaq_requires_int8_mode(self):
        """aaq raises EmulatorError when not in INT8 mode."""
        from ipu_emu.ipu import EmulatorError
        state = _make_state("aaq 0;;\nBKPT;;")
        state.dtype = DType.E4
        with pytest.raises(EmulatorError, match="INT8 mode"):
            run_until_complete(state)

    def test_aaq_full_xmem_row_1_ignores_valid_elements(self):
        """full_xmem_row=1 always quantizes all 128 lanes even if CR15.valid_elements < 128."""
        state = IpuState()
        state.dtype = DType.INT8
        state.set_cr_dstructure(valid_elements=64)
        self._set_acc_words(state, [1] * 128)  # in-range; direct clamp leaves 1 -> 1

        encoded = assemble("aaq 1;;\nBKPT;;")
        from ipu_emu.execute import decode_instruction_word
        from ipu_emu.emulator import load_program, run_until_complete
        decoded = [decode_instruction_word(w) for w in encoded]
        load_program(state, decoded)
        run_until_complete(state)

        result = state.regfile.get_post_aaq_reg()
        assert result[:128] == bytearray([1] * 128), "all 128 lanes should be quantized"
        assert result[128:] == bytearray(384)

    def test_aaq_full_xmem_row_0_uses_valid_elements(self):
        """full_xmem_row=0 quantizes only CR15.valid_elements lanes; rest are zeroed."""
        state = IpuState()
        state.dtype = DType.INT8
        state.set_cr_dstructure(valid_elements=48)
        self._set_acc_words(state, [1] * 128)  # in-range; direct clamp leaves 1 -> 1

        encoded = assemble("aaq 0;;\nBKPT;;")
        from ipu_emu.execute import decode_instruction_word
        from ipu_emu.emulator import load_program, run_until_complete
        decoded = [decode_instruction_word(w) for w in encoded]
        load_program(state, decoded)
        run_until_complete(state)

        result = state.regfile.get_post_aaq_reg()
        assert result[:48] == bytearray([1] * 48), "first 48 lanes should be quantized"
        assert result[48:] == bytearray(464), "remaining bytes should be zero"

    def test_str_post_aaq_reg_int8_writes_leading_128_to_xmem(self):
        """INT8 mode: STR_POST_AAQ_REG stores only the 128 packed INT8 bytes.

        Full path: AAQ packs the INT8 lanes into post_aaq_reg[0:128], then
        STR_POST_AAQ_REG drains exactly those 128 bytes so they round-trip as a
        128-byte input chunk.
        """
        state = IpuState()
        state.dtype = DType.INT8
        # Wide int32 lanes spanning the clamp range; AAQ clamps to [-128, 127].
        self._set_acc_words(state, [i - 64 for i in range(128)])
        state.regfile.set_cr(8, 0x4000)

        encoded = assemble(
            """\
SET lr0 cr8;;
aaq 1;;
str_post_aaq_reg lr0 cr0;;
BKPT;;
"""
        )
        from ipu_emu.execute import decode_instruction_word
        from ipu_emu.emulator import load_program, run_until_complete
        decoded = [decode_instruction_word(w) for w in encoded]
        load_program(state, decoded)
        run_until_complete(state)

        # Exactly 128 INT8 bytes written; nothing beyond.
        stored = state.xmem.read_address(0x4000, 128)
        assert stored == bytearray((i - 64) & 0xFF for i in range(128))
        assert state.xmem.read_address(0x4000 + 128, 384) == bytearray(384)

    def test_str_post_aaq_reg_wide_mode_writes_full_512_to_xmem(self):
        """Wide-vector (FP32) mode: STR_POST_AAQ_REG stores the full 512 bytes."""
        from ipu_emu.execute import decode_instruction_word
        from ipu_emu.emulator import load_program, run_until_complete
        from ipu_emu.ipu_state import WideVectorArithmetic

        state = IpuState(
            wide_vector_debug=True,
            wide_vector_arithmetic=WideVectorArithmetic.FP32,
        )
        state.dtype = DType.INT8  # mode byte; wide flag overrides element width
        # Distinct FP32 lanes across all 512 bytes (no zero pad).
        post = struct.pack("<128f", *[float(i + 1) for i in range(128)])
        state.regfile.set_post_aaq_reg(bytearray(post))
        state.regfile.set_cr(8, 0x4000)

        encoded = assemble("SET lr0 cr8;;\nstr_post_aaq_reg lr0 cr0;;\nBKPT;;")
        decoded = [decode_instruction_word(w) for w in encoded]
        load_program(state, decoded)
        run_until_complete(state)

        stored = state.xmem.read_address(0x4000, 512)
        assert stored == state.regfile.get_post_aaq_reg()

    def test_STR_ACC_REG_emits_warning(self):
        """STR_ACC_REG emits a UserWarning about being debug-only."""
        state = IpuState()
        state.dtype = DType.INT8

        encoded = assemble("STR_ACC_REG lr0 cr0;;\nBKPT;;")
        from ipu_emu.execute import decode_instruction_word
        from ipu_emu.emulator import load_program, run_until_complete
        decoded = [decode_instruction_word(w) for w in encoded]
        load_program(state, decoded)

        with pytest.warns(UserWarning, match="DEBUG ONLY"):
            run_until_complete(state)


# ============================================================================
# ACTIVATE (element-wise: r_acc → POST_AAQ_REG, issue #77)
# ============================================================================


def _post_aaq_lane_i32(state: IpuState, lane: int) -> int:
    buf = state.regfile.get_post_aaq_reg()
    word = struct.unpack_from("<I", buf, lane * 4)[0]
    return struct.unpack("<i", struct.pack("<I", word))[0]


def _post_aaq_lane_f32(state: IpuState, lane: int) -> float:
    buf = state.regfile.get_post_aaq_reg()
    word = struct.unpack_from("<I", buf, lane * 4)[0]
    return struct.unpack("<f", struct.pack("<I", word))[0]


class TestActivate:
    """ACTIVATE applies ipu_common.activations to r_acc lanes; results go to POST_AAQ_REG."""

    def test_activate_relu_int32(self):
        state = _make_state(
            """\
ACTIVATE relu 0;;
BKPT;;
"""
        )
        state.dtype = DType.INT8
        state.set_cr_dstructure(128)
        state.regfile.set_r_acc_word(
            0, struct.unpack("<I", struct.pack("<i", -9))[0]
        )
        run_until_complete(state)
        assert (
            struct.unpack("<i", struct.pack("<I", state.regfile.get_r_acc_word(0)))[0]
            == -9
        )
        assert _post_aaq_lane_i32(state, 0) == 0

    def test_activate_masks_inactive_lanes(self):
        state = _make_state(
            """\
ACTIVATE relu 0;;
BKPT;;
"""
        )
        state.dtype = DType.INT8
        state.set_cr_dstructure(2)
        state.regfile.set_r_acc_word(
            0, struct.unpack("<I", struct.pack("<i", -5))[0]
        )
        state.regfile.set_r_acc_word(
            1, struct.unpack("<I", struct.pack("<i", -3))[0]
        )
        sentinel = struct.unpack("<I", struct.pack("<i", 99_999))[0]
        for i in range(2, 128):
            state.regfile.set_r_acc_word(i, sentinel)
        post = bytearray(512)
        for i in range(2, 128):
            struct.pack_into("<i", post, i * 4, 99_999)
        state.regfile.set_post_aaq_reg(post)
        run_until_complete(state)
        assert (
            struct.unpack("<i", struct.pack("<I", state.regfile.get_r_acc_word(0)))[0]
            == -5
        )
        assert (
            struct.unpack("<i", struct.pack("<I", state.regfile.get_r_acc_word(1)))[0]
            == -3
        )
        for i in range(2, 128):
            assert state.regfile.get_r_acc_word(i) == sentinel
        assert _post_aaq_lane_i32(state, 0) == 0
        assert _post_aaq_lane_i32(state, 1) == 0
        for i in range(2, 128):
            assert _post_aaq_lane_i32(state, i) == 99_999

    def test_activate_identity_keyword_is_noop(self):
        state = _make_state(
            """\
ACTIVATE identity 0;;
BKPT;;
"""
        )
        state.dtype = DType.INT8
        state.set_cr_dstructure(1)
        raw = struct.unpack("<I", struct.pack("<i", -42))[0]
        state.regfile.set_r_acc_word(0, raw)
        run_until_complete(state)
        assert state.regfile.get_r_acc_word(0) == raw
        assert _post_aaq_lane_i32(state, 0) == struct.unpack("<i", struct.pack("<I", raw))[0]

    def test_activate_sigmoid_float_lane(self):
        state = _make_state(
            """\
ACTIVATE sigmoid 0;;
BKPT;;
"""
        )
        state.dtype = DType.E4
        state.set_cr_dstructure(1)
        state.regfile.set_r_acc_word(
            0, struct.unpack("<I", struct.pack("<f", 0.0))[0]
        )
        run_until_complete(state)
        out = _post_aaq_lane_f32(state, 0)
        assert abs(out - 0.5) < 1e-6

    def test_activate_matches_reference_all_ids_float(self):
        x = 0.25
        for fid, name in enumerate(ACTIVATION_FN_NAMES):
            state = _make_state(
                f"""\
ACTIVATE {name} 0;;
BKPT;;
"""
            )
            state.dtype = DType.E4
            state.set_cr_dstructure(1)
            state.regfile.set_r_acc_word(
                0, struct.unpack("<I", struct.pack("<f", x))[0]
            )
            run_until_complete(state)
            got = _post_aaq_lane_f32(state, 0)
            exp = apply_activation(fid, x)
            assert abs(got - exp) < 1e-5, f"id={fid} got={got} exp={exp}"

    def test_activate_exp2_float(self):
        state = _make_state(
            """\
ACTIVATE exp2 0;;
BKPT;;
"""
        )
        state.dtype = DType.E4
        state.set_cr_dstructure(1)
        state.regfile.set_r_acc_word(
            0, struct.unpack("<I", struct.pack("<f", 3.0))[0]
        )
        run_until_complete(state)
        out = _post_aaq_lane_f32(state, 0)
        assert abs(out - 8.0) < 1e-5

    def test_activate_gelu_float(self):
        state = _make_state(
            """\
ACTIVATE gelu 0;;
BKPT;;
"""
        )
        state.dtype = DType.E4
        state.set_cr_dstructure(1)
        x = 1.0
        state.regfile.set_r_acc_word(
            0, struct.unpack("<I", struct.pack("<f", x))[0]
        )
        run_until_complete(state)
        out = _post_aaq_lane_f32(state, 0)
        assert abs(out - apply_activation(5, x)) < 1e-5

    def test_activate_elu_respects_ipu_state_alpha(self):
        """``IpuState`` α overrides module defaults for ``ACTIVATE`` (not CR)."""
        x = -1.0
        alpha = 0.5
        state = _make_state(
            """\
ACTIVATE elu 0;;
BKPT;;
""",
            elu_alpha=alpha,
        )
        state.dtype = DType.E4
        state.set_cr_dstructure(1)
        state.regfile.set_r_acc_word(
            0, struct.unpack("<I", struct.pack("<f", x))[0]
        )
        run_until_complete(state)
        out = _post_aaq_lane_f32(state, 0)
        exp = apply_activation(7, x, elu_alpha=alpha)
        assert abs(out - exp) < 1e-5

    def test_activate_elu_after_set_activation_alphas(self):
        x = -2.0
        alpha = 0.125
        state = _make_state(
            """\
ACTIVATE elu 0;;
BKPT;;
"""
        )
        state.set_activation_alphas(elu_alpha=alpha)
        state.dtype = DType.E4
        state.set_cr_dstructure(1)
        state.regfile.set_r_acc_word(
            0, struct.unpack("<I", struct.pack("<f", x))[0]
        )
        run_until_complete(state)
        out = _post_aaq_lane_f32(state, 0)
        exp = apply_activation(7, x, elu_alpha=alpha)
        assert abs(out - exp) < 1e-5

    def test_activate_valid_elements_from_cr15(self):
        state = _make_state(
            """\
ACTIVATE relu 0;;
BKPT;;
"""
        )
        state.dtype = DType.INT8
        state.set_cr_dstructure(1)
        state.regfile.set_r_acc_word(
            0, struct.unpack("<I", struct.pack("<i", -8))[0]
        )
        tail = struct.unpack("<I", struct.pack("<i", 123))[0]
        state.regfile.set_r_acc_word(1, tail)
        run_until_complete(state)
        assert (
            struct.unpack("<i", struct.pack("<I", state.regfile.get_r_acc_word(0)))[0]
            == -8
        )
        assert state.regfile.get_r_acc_word(1) == tail
        assert _post_aaq_lane_i32(state, 0) == 0
        assert _post_aaq_lane_i32(state, 1) == 0

    def test_activate_reciprocal_float(self):
        state = _make_state(
            """\
ACTIVATE reciprocal 0;;
BKPT;;
"""
        )
        state.dtype = DType.E4
        state.set_cr_dstructure(1)
        x = 4.0
        state.regfile.set_r_acc_word(
            0, struct.unpack("<I", struct.pack("<f", x))[0]
        )
        run_until_complete(state)
        out = _post_aaq_lane_f32(state, 0)
        assert abs(out - 0.25) < 1e-6

    def test_activate_reciprocal_zero_input(self):
        state = _make_state(
            """\
ACTIVATE reciprocal 0;;
BKPT;;
"""
        )
        state.dtype = DType.E4
        state.set_cr_dstructure(1)
        state.regfile.set_r_acc_word(
            0, struct.unpack("<I", struct.pack("<f", 0.0))[0]
        )
        run_until_complete(state)
        out = _post_aaq_lane_f32(state, 0)
        assert out == 0.0

    def test_activate_rsqrt_float(self):
        state = _make_state(
            """\
ACTIVATE rsqrt 0;;
BKPT;;
"""
        )
        state.dtype = DType.E4
        state.set_cr_dstructure(1)
        x = 4.0
        state.regfile.set_r_acc_word(
            0, struct.unpack("<I", struct.pack("<f", x))[0]
        )
        run_until_complete(state)
        out = _post_aaq_lane_f32(state, 0)
        assert abs(out - 0.5) < 1e-6

    def test_activate_rsqrt_nonpositive_input(self):
        state = _make_state(
            """\
ACTIVATE rsqrt 0;;
BKPT;;
"""
        )
        state.dtype = DType.E4
        state.set_cr_dstructure(1)
        state.regfile.set_r_acc_word(
            0, struct.unpack("<I", struct.pack("<f", 0.0))[0]
        )
        run_until_complete(state)
        out = _post_aaq_lane_f32(state, 0)
        assert out == 0.0

    def test_activate_full_xmem_row_1_ignores_valid_elements(self):
        """ACTIVATE full_xmem_row=1: activates all 128 lanes even when CR15.valid_elements < 128."""
        state = _make_state(
            """\
ACTIVATE relu 1;;
BKPT;;
"""
        )
        state.dtype = DType.INT8
        state.set_cr_dstructure(valid_elements=4)
        for i in range(128):
            state.regfile.set_r_acc_word(i, struct.unpack("<I", struct.pack("<i", i + 1))[0])
        run_until_complete(state)
        for i in range(128):
            assert _post_aaq_lane_i32(state, i) == i + 1, f"lane {i} should be activated"

    def test_activate_full_xmem_row_0_uses_valid_elements(self):
        """ACTIVATE full_xmem_row=0: activates only CR15.valid_elements lanes; rest unchanged."""
        state = _make_state(
            """\
ACTIVATE relu 0;;
BKPT;;
"""
        )
        state.dtype = DType.INT8
        state.set_cr_dstructure(valid_elements=4)
        for i in range(128):
            state.regfile.set_r_acc_word(i, struct.unpack("<I", struct.pack("<i", i + 1))[0])
        run_until_complete(state)
        for i in range(4):
            assert _post_aaq_lane_i32(state, i) == i + 1, f"lane {i} should be activated"
        for i in range(4, 128):
            assert _post_aaq_lane_i32(state, i) == 0, f"lane {i} should be untouched (zero)"
