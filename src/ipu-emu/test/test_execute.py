"""Tests for the VLIW execution engine.

These tests use the assembler (``ipu_as.lark_tree.assemble``) to turn
inline assembly into encoded instruction words, load them into an
``IpuState``, execute, and check results.
"""

from __future__ import annotations

import math
import os
import struct
import subprocess
import sys

import pytest
from emu_support import (
    E4,
    BreakpointRunControl,
    ElementFormat,
    decode_program,
    make_state,
    new_state,
    run,
    write_nx8_row,
)

from ipu_as.compound_inst import CompoundInst
from ipu_as.lark_tree import assemble, parse
from ipu_emu.activations import AaqOptions, ActivationParams, apply_activation
from ipu_emu.emulator import (
    DebugAction,
    load_program,
    run_until_complete,
    run_with_debug,
)
from ipu_emu.errors import ConfigurationError, CrContentError, EmulatorFault
from ipu_emu.ipu import EIGHT_BIT_ROWS, EmulatorError, Ipu
from ipu_emu.ipu_state import HaltRecord, IpuState
from ipu_emu.outcome import CycleOutcome
from ipu_emu.xmem import XMEM_SIZE_BYTES
from ipu_isa.codec.word import DecodedWord
from ipu_isa.cr_halves import MaskType, constant, ds, qf
from ipu_isa.faults import FaultId
from ipu_isa.nx8 import Nx8Format, decode_row, encode_row, split_metadata


def int_constant(value: float) -> int:
    """A MULT constant CR holding *value* (one signed fe-4 NX8 element, ADR 0007)."""
    row = encode_row([float(value)], E4)
    return constant(row.elements[0], row.scale, E4.exp_format, E4.sign)


def fill_nx8_cyclic(state: IpuState, values) -> None:
    """Fill R_CYCLIC (512 elements) with *values*, quantized row by row, and each slot's metadata."""
    values = list(values) + [0] * (512 - len(values))
    for k in range(4):
        row = encode_row(values[k * 128 : (k + 1) * 128], E4)
        state.regfile.set_r_cyclic_at(k * 128, row.elements)
        state.regfile.set_register_bytes("r_cyclic_meta", k, row.metadata.to_bytes(2, "little"))


def fill_nx8_ra(state: IpuState, values) -> None:
    """Fill R0 ++ R1 (256 elements) with *values*, quantized row by row, and their metadata."""
    values = list(values) + [0] * (256 - len(values))
    for k in range(2):
        row = encode_row(values[k * 128 : (k + 1) * 128], E4)
        state.regfile.set_register_bytes("r", k, row.elements)
        state.regfile.set_register_bytes("r_meta", k, row.metadata.to_bytes(2, "little"))


def acc_lanes(state: IpuState) -> tuple[float, ...]:
    """R_ACC as 128 FP32 values."""
    return struct.unpack("<128f", state.regfile.get_r_acc_bytes())


# ============================================================================
# Basic Register Operations
# ============================================================================


class TestRegisterOperations:
    def test_set_lr(self):
        state = run("SET lr13 cr8;;\nHALT;;", cr={8: 0x1000})
        assert state.regfile.get_lr(13) == 0x1000

    def test_add_imm_accumulates_lr(self):
        state = run(
            """\
SET lr11 cr8;;
INC lr11 5;;
INC lr11 3;;
HALT;;
""",
            cr={8: 10},
        )
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
        state = run(
            """\
SET lr1 cr8;;
SET lr2 cr9;;
ADD lr3 lr1 lr2;;
HALT;;
""",
            cr={8: 100, 9: 50},
        )
        assert state.regfile.get_lr(1) == 100
        assert state.regfile.get_lr(2) == 50
        assert state.regfile.get_lr(3) == 150

    def test_add_lr_cr(self):
        state = make_state(
            """\
SET lr1 cr8;;
ADD lr4 lr1 cr5;;
HALT;;
""",
            cr={8: 200},
        )
        state.regfile.set_cr(5, 75)
        run_until_complete(state)
        assert state.regfile.get_lr(1) == 200
        assert state.regfile.get_cr(5) == 75
        assert state.regfile.get_lr(4) == 275

    def test_add_lr_lr_imm5(self):
        state = run(
            """\
SET lr1 cr8;;
SET lr2 cr9;;
ADD lr4 lr1 lr2;;
HALT;;
""",
            cr={8: 200, 9: 11},
        )
        assert state.regfile.get_lr(4) == 211

    def test_sub_lr_lr(self):
        state = run(
            """\
SET lr1 cr8;;
SET lr2 cr9;;
SUB lr3 lr1 lr2;;
HALT;;
""",
            cr={8: 100, 9: 30},
        )
        assert state.regfile.get_lr(3) == 70

    def test_sub_lr_lr_cr(self):
        state = make_state(
            """\
SET lr2 cr8;;
SUB lr5 lr2 cr3;;
HALT;;
""",
            cr={8: 45},
        )
        state.regfile.set_cr(3, 200)
        run_until_complete(state)
        assert state.regfile.get_lr(5) == (45 - 200) & 0xFFFFFFFF

    def test_sub_lr_lr_imm5(self):
        state = run(
            """\
SET lr2 cr8;;
SET lr4 cr9;;
SUB lr3 lr2 lr4;;
HALT;;
""",
            cr={8: 100, 9: 30},
        )
        assert state.regfile.get_lr(3) == 70


# ============================================================================
# INC.MODP2 (Issue #47): dst <- (dst + step) mod 2^k
# ============================================================================


class TestIncrModPow2:
    def test_basic_wrap_small_k(self):
        """k=4 → mod 16; 15 + 1 wraps to 0."""
        state = run(
            """\
SET lr0 cr8;;
SET lr1 cr9;;
INC.MODP2 lr0 lr1 4;;
HALT;;
""",
            cr={8: 15, 9: 1},
        )
        assert state.regfile.get_lr(0) == 0

    def test_k9_mask_511(self):
        """k=9 → mask 511; largest legal k from spec."""
        state = run(
            """\
SET lr0 cr8;;
SET lr1 cr9;;
INC.MODP2 lr0 lr1 9;;
HALT;;
""",
            cr={8: 500, 9: 20},
        )
        assert state.regfile.get_lr(0) == (500 + 20) & 511

    def test_read_before_write_same_register(self):
        """dst and step both lr0: uses snapshot value of lr0 for the sum."""
        state = run(
            """\
SET lr0 cr8;;
INC.MODP2 lr0 lr0 3;;
HALT;;
""",
            cr={8: 5},
        )
        assert state.regfile.get_lr(0) == (5 + 5) & 7

    def test_step_from_cr(self):
        state = make_state(
            """\
SET lr0 cr8;;
INC.MODP2 lr0 cr4 4;;
HALT;;
""",
            cr={8: 2},
        )
        state.regfile.set_cr(4, 10)
        run_until_complete(state)
        assert state.regfile.get_lr(0) == (2 + 10) & 15

    def test_large_unsigned_step_reduces_mod_mask(self):
        """Uint32 add before mask: step loaded from CR as full uint32."""
        state = make_state(
            """\
SET lr0 cr8;;
INC.MODP2 lr0 cr5 3;;
HALT;;
""",
            cr={8: 3},
        )
        state.regfile.set_cr(5, 0xFFFFFFFE)
        run_until_complete(state)
        assert state.regfile.get_lr(0) == (3 + 0xFFFFFFFE) & 7

    def test_k_encoded_four_bits(self):
        """k operand uses 4 bits: semantic k=9 encodes as 8 in the instruction word."""
        encoded = assemble("INC.MODP2 lr0 lr1 9;; HALT;;")
        d = DecodedWord.from_word(encoded[0])
        assert d["lr_inst_0_token_3_lr_reg_field"] == 8

    def test_assembler_rejects_k_out_of_range(self):
        with pytest.raises(ValueError, match=r"INC.MODP2 k operand"):
            CompoundInst(parse("INC.MODP2 lr0 lr1 0;;")[0])
        with pytest.raises(ValueError, match=r"INC.MODP2 k operand"):
            CompoundInst(parse("INC.MODP2 lr0 lr1 10;;")[0])

    def test_two_incr_mod_pow2_different_dst_no_conflict(self):
        """Two INC.MODP2 writing different LRs must not raise a conflict error."""
        state = run(
            """\
SET lr5 cr8;;
SET lr12 cr9;;
SET lr13 cr10;;
INC.MODP2 lr5 lr12 7; INC.MODP2 lr6 lr13 8;;
HALT;;
""",
            cr={8: 0, 9: 1, 10: 128},
        )
        assert state.regfile.get_lr(5) == 1
        assert state.regfile.get_lr(6) == 128


class TestIncDec:
    def test_inc_dest(self):
        state = run(
            """\
SET lr5 cr8;;
INC lr5 7;;
HALT;;
""",
            cr={8: 100},
        )
        assert state.regfile.get_lr(5) == 107

    def test_dec_dest(self):
        state = run(
            """\
SET lr5 cr8;;
DEC lr5 30;;
HALT;;
""",
            cr={8: 100},
        )
        assert state.regfile.get_lr(5) == 70

    def test_inc_dec_same_cycle(self):
        state = run(
            """\
SET lr0 cr8;;
SET lr1 cr9;;
INC lr0 5;;
DEC lr1 2;;
HALT;;
""",
            cr={8: 10, 9: 20},
        )
        assert state.regfile.get_lr(0) == 15
        assert state.regfile.get_lr(1) == 18

    def test_decode_inc_imm_operand_field(self):
        """``INC`` immediate uses the union-derived LrIncDecImmediate field."""
        encoded = assemble("INC lr2 7;; HALT;;")
        d = DecodedWord.from_word(encoded[0])
        assert d["lr_inst_0_token_0_lr_inst_opcode"] == 4  # inc
        assert d["lr_inst_0_token_2_lr_reg_field"] == 2
        assert (
            d["lr_inst_0_token_1_addbi_immediate"] == 7
        )  # imm in field shared with AddbiImmediate


class TestAddb:
    """ADDB/ADDBI: broadcast-add a signed byte to LRDn's 8 byte elements, saturating to [0, 255]."""

    def test_addb_lr_source(self):
        """Register source: each of the 8 elements across LR0/LR1 gains the same byte."""
        state = run(
            """\
SET lr0 cr8;;
SET lr1 cr9;;
SET lr5 cr10;;
ADDB lrd0 lr5;;
HALT;;
""",
            cr={8: 0x01020304, 9: 0x05060708, 10: 10},
        )
        assert state.regfile.get_lr(0) == 0x0B0C0D0E
        assert state.regfile.get_lr(1) == 0x0F101112

    def test_addb_cr_source(self):
        """CR source: src_b's low byte is read directly from the CR (no intermediate SET)."""
        state = run(
            """\
SET lr0 cr8;;
SET lr1 cr9;;
ADDB lrd0 cr11;;
HALT;;
""",
            cr={8: 0x01020304, 9: 0x05060708, 11: 10},
        )
        assert state.regfile.get_lr(0) == 0x0B0C0D0E
        assert state.regfile.get_lr(1) == 0x0F101112

    def test_addbi_immediate(self):
        """Immediate source: unsigned 200 reinterprets as signed -56, decreasing each element."""
        state = run(
            """\
SET lr0 cr8;;
SET lr1 cr9;;
ADDBI lrd0 200;;
HALT;;
""",
            cr={8: 0x64646464, 9: 0x64646464},
        )
        # 0x64 == 100 in every lane; 100 + (-56) == 44 == 0x2C in every lane.
        assert state.regfile.get_lr(0) == 0x2C2C2C2C
        assert state.regfile.get_lr(1) == 0x2C2C2C2C

    def test_addbi_negative_literal_matches_unsigned_spelling(self):
        """``-56`` and ``200`` encode identically and produce the same result."""
        state = run(
            """\
SET lr0 cr8;;
ADDBI lrd0 -56;;
HALT;;
""",
            cr={8: 0x64646464},
        )
        assert state.regfile.get_lr(0) == 0x2C2C2C2C

    def test_addbi_saturates_at_255_ceiling(self):
        """An element already near 255 clamps at 255 instead of wrapping."""
        state = run(
            """\
SET lr0 cr8;;
ADDBI lrd0 20;;
HALT;;
""",
            cr={8: 250},
        )  # LR0 = 0x000000FA: low lane 250, others 0
        lanes = state.regfile.get_lr(0).to_bytes(4, "little")
        assert lanes[0] == 255  # 250 + 20 saturates at 255
        assert lanes[1] == 20  # 0 + 20, no saturation

    def test_addbi_saturates_at_0_floor(self):
        """A negative byte pushes a low element below 0, clamping at 0 instead of wrapping."""
        state = run(
            """\
SET lr0 cr8;;
ADDBI lrd0 200;;
HALT;;
""",
            cr={8: 5},
        )  # LR0 = 0x00000005; 200 decodes as signed -56
        assert state.regfile.get_lr(0) == 0  # every lane clamps to 0 (5-56 and 0-56 both < 0)
        assert state.regfile.get_lr(1) == 0

    def test_addb_lrd_maps_to_correct_lr_pair(self):
        """LRD6 = LR7:LR6 (named after the lower register) — only LR6/LR7 change; other LRs are untouched."""
        state = run(
            """\
SET lr6 cr8;;
SET lr7 cr9;;
ADDBI lrd6 5;;
HALT;;
""",
            cr={8: 100, 9: 50},
        )
        # +5 broadcasts to all 8 lanes, including LR6/LR7's zero upper bytes.
        assert state.regfile.get_lr(6) == 84215145  # bytes [105, 5, 5, 5]
        assert state.regfile.get_lr(7) == 84215095  # bytes [55, 5, 5, 5]
        assert state.regfile.get_lr(0) == 0
        assert state.regfile.get_lr(5) == 0

    def test_addb_lrd_conflict_with_overlapping_lr_write(self):
        """ADDBI on LRD0 (LR0/LR1) conflicts with a same-cycle plain write to LR1."""
        with pytest.raises(EmulatorFault, match="LR conflict") as caught:
            run("ADDBI lrd0 5; INC lr1 3;;\nHALT;;\n")
        assert (caught.value.fault, caught.value.pc) == (FaultId.F3, 0)

    def test_decode_addb_operand_fields(self):
        """``ADDB``'s dest/src_b share the same union fields as INC/SET.

        ``lrd4`` is the 3rd pair (LR4/LR5), so it encodes as index 2.
        """
        encoded = assemble("ADDB lrd4 lr5;; HALT;;")
        d = DecodedWord.from_word(encoded[0])
        assert d["lr_inst_0_token_0_lr_inst_opcode"] == 6  # addb
        assert d["lr_inst_0_token_2_lr_reg_field"] == 2  # dest = lrd4 (pair index 2)
        assert d["lr_inst_0_token_1_addbi_immediate"] == 5  # src_b = lr5

    def test_decode_addbi_operand_fields(self):
        """``ADDBI``'s immediate uses the same shared field as INC/DEC.

        ``lrd8`` is the 5th pair (LR8/LR9), so it encodes as index 4.
        """
        encoded = assemble("ADDBI lrd8 200;; HALT;;")
        d = DecodedWord.from_word(encoded[0])
        assert d["lr_inst_0_token_0_lr_inst_opcode"] == 7  # addbi
        assert d["lr_inst_0_token_2_lr_reg_field"] == 4  # dest = lrd8 (pair index 4)
        assert d["lr_inst_0_token_1_addbi_immediate"] == 200


# ============================================================================
# Three LR sub-slots per VLIW (SLOT_COUNT["lr"] == 3)
# ============================================================================


class TestThreeLrSlots:
    def test_three_lr_instructions_parallel(self):
        """Three independent LR ops may execute in the same cycle."""
        state = run(
            """\
SET lr0 cr8; SET lr1 cr9; SET lr2 cr10;;
HALT;;
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
        encoded = assemble("SET lr4 cr8; SET lr5 cr9; SET lr6 cr10;;\nHALT;;")
        assert len(encoded) == 2
        d = DecodedWord.from_word(encoded[0])
        assert d["lr_inst_0_token_0_lr_inst_opcode"] == 0  # set
        assert d["lr_inst_0_token_2_lr_reg_field"] == 4  # reg = lr4
        assert (
            d["lr_inst_0_token_1_addbi_immediate"] == 16 + 8
        )  # src = cr8 (LcrHalfIdx: CRs follow the 16 LRs)
        assert d["lr_inst_0_token_3_lr_reg_field"] == 0  # unused field (default lr0)
        assert d["lr_inst_1_token_2_lr_reg_field"] == 5  # reg = lr5
        assert d["lr_inst_1_token_1_addbi_immediate"] == 16 + 9  # src = cr9
        assert d["lr_inst_1_token_3_lr_reg_field"] == 0  # unused field (default lr0)
        assert d["lr_inst_2_token_2_lr_reg_field"] == 6  # reg = lr6
        assert d["lr_inst_2_token_1_addbi_immediate"] == 16 + 10  # src = cr10
        assert d["lr_inst_2_token_3_lr_reg_field"] == 0  # unused field (default lr0)

    def test_decode_add_lcr_operand_field(self):
        """``ADD`` third operand uses LcrRegField (register-only), sharing a field with AddbiImmediate."""
        encoded = assemble("ADD lr2 lr1 cr7;; HALT;;")
        d = DecodedWord.from_word(encoded[0])
        assert d["lr_inst_0_token_0_lr_inst_opcode"] == 1  # add
        assert d["lr_inst_0_token_2_lr_reg_field"] == 2  # dest = lr2
        assert d["lr_inst_0_token_3_lr_reg_field"] == 1  # src_a = lr1
        assert d["lr_inst_0_token_1_addbi_immediate"] == 16 + 7  # src_b = cr7


# ============================================================================
# Memory Operations
# ============================================================================


class TestMemoryOperations:
    def test_load_from_memory(self):
        """cr8=32 is a row number: row 32 * 128 B/row (NX8) = byte 0x1000; the
        row's metadata word lands in r_meta beside it (ADR 0007)."""
        test_data = bytes(range(128))
        state = make_state(
            """\
SET lr13 cr8;;
LD.MULT r1 lr13 cr0;;
HALT;;
""",
            cr={8: 32},
        )
        state.xmem.write_address(0x1000, test_data)
        state.xmem.write_metadata(0x1000, 0x185)
        run_until_complete(state)

        assert state.regfile.get_lr(13) == 32
        assert state.regfile.get_r(1) == bytearray(test_data)
        assert bytes(state.regfile.view("r_meta")) == b"\x00\x00\x85\x01"

    def test_fp32_load_writes_zero_metadata(self):
        """An FP32 row is 512 B with no metadata: LD.MULT writes 0 to r_meta."""
        state = make_state("LD.MULT r1 lr0 cr2;;\nHALT;;", cr={2: 3}, fmt=ElementFormat.FP32)
        state.regfile.set_register_bytes("r_meta", 1, b"\xff\xff")
        state.xmem.write_address(3 * 512, struct.pack("<128f", *([2.5] * 128)))
        run_until_complete(state)
        assert state.regfile.get_r_wide_debug(1) == bytearray(struct.pack("<128f", *([2.5] * 128)))
        assert bytes(state.regfile.view("r_meta")) == bytes(4)

    def test_multiply_accumulate(self):
        """NX8: r1 = all-2, cyclic = all-3, MULT.RC.VV -> each R_ACC lane is 6.0."""
        state = make_state(
            """\
SET lr13 cr8;;
LD.MULT r1 lr13 cr0;;
SET lr14 cr9;;
SET lr15 cr10;;
LD.MULT.CYCLIC lr14 cr0 lr15;;
MULT.RC.VV lr0 r1 0 lr0 cr15.l cr14.l;
ACC.ADD;;
HALT;;
""",
            cr={8: 32, 9: 64, 10: 0},
        )
        write_nx8_row(state, 32, [2] * 128)
        write_nx8_row(state, 64, [3] * 128)
        run_until_complete(state)

        words = struct.unpack_from("<128f", state.regfile.get_r_acc_bytes())
        assert words == (6.0,) * 128

    def test_cyclic_register_load(self):
        cyclic_data = bytes([(i * 2) & 0xFF for i in range(128)])
        state = make_state(
            """\
SET lr0 cr8;;
SET lr1 cr9;;
LD.MULT.CYCLIC lr0 cr0 lr1;;
HALT;;
""",
            cr={8: 160, 9: 256},
        )
        state.xmem.write_address(0x5000, cyclic_data)
        state.xmem.write_metadata(0x5000, 0x7F)
        run_until_complete(state)

        assert state.regfile.get_lr(0) == 160
        assert state.regfile.get_r_cyclic_at(256, 128) == bytearray(cyclic_data)
        assert bytes(state.regfile.view("r_cyclic_meta")) == bytes(4) + b"\x7f\x00" + bytes(2)

    def test_cyclic_register_load_invalid_index_raises(self):
        """index must be one of the four R_CYCLIC slot boundaries — no implicit wrap."""
        state = make_state(
            """\
SET lr0 cr8;;
SET lr1 cr9;;
LD.MULT.CYCLIC lr0 cr0 lr1;;
HALT;;
""",
            cr={8: 160, 9: 64},
        )
        with pytest.raises(EmulatorFault, match="index must be one of") as caught:
            run_until_complete(state)
        assert state.halt == HaltRecord("fault", caught.value.pc, caught.value.fault)
        assert caught.value.fault is FaultId.F4

    def test_mask_register_load(self):
        mask_data = bytes([(i + 1) & 0xFF for i in range(128)])
        state = make_state(
            """\
SET lr0 cr8;;
LD.MASK lr0 cr0;;
HALT;;
""",
            cr={8: 192},
        )
        state.xmem.write_address(0x6000, mask_data)
        run_until_complete(state)

        assert state.regfile.get_lr(0) == 192
        assert state.regfile.get_r_mask() == bytearray(mask_data)

    def test_concurrent_load_and_store_same_cycle(self):
        """Load and AaQ store in one VLIW bundle: the load reads the row before
        the store (AAQ runs after LOAD) overwrites it and its metadata."""
        row = 32
        old_data = bytes([i & 0xFF for i in range(128)])
        state = make_state(
            """\
SET lr0 cr8;;
LD.MULT r0 lr0 cr0 ; ACTIVATE.QUANTIZE identity cr15.l cr14.l cr8 lr1;;
HALT;;
""",
            cr={8: row},
        )
        state.xmem.write_address(row * 128, old_data)
        state.regfile.set_r_acc_bytes(struct.pack("<128f", *([4.0] * 128)))
        run_until_complete(state)

        assert state.regfile.get_r(0) == bytearray(old_data)  # pre-store memory
        assert state.xmem.read_address(row * 128, 128) != old_data  # the store landed
        assert state.xmem.read_metadata(row * 128) != 0

    @staticmethod
    def _masked(  # noqa: PLR0913 -- a MULT.RC.VV case: mask, DS, mode and operands
        mask_data: bytes,
        mask_offset: int,
        shift_cr: int,
        *,
        ds_half: int | None = None,
        fmt: ElementFormat = ElementFormat.NX8,
        r0=2,
        rc=3,
    ) -> tuple[float, ...]:
        """MULT.RC.VV of r0 x rc under a mask; the MULT_RES lanes as FP32."""
        state = make_state(
            f"""\
SET lr0 cr8;;
LD.MULT r0 lr0 cr0;;
SET lr1 cr9;;
SET lr2 cr10;;
LD.MULT.CYCLIC lr1 cr0 lr2;;
SET lr3 cr11;;
LD.MASK lr3 cr0;;
SET lr5 cr12;;
SET lr6 cr10;;
MULT.RC.VV lr6 r0 {mask_offset} lr5 cr15.l cr14.l;
ACC.ADD;;
HALT;;
""",
            cr={8: 32, 9: 64, 10: 0, 11: 96, 12: shift_cr},
            fmt=fmt,
            ds=ds_half,
        )
        if fmt is ElementFormat.FP32:
            state.xmem.write_address(32 * 512, struct.pack("<128f", *([float(r0)] * 128)))
            state.xmem.write_address(64 * 512, struct.pack("<128f", *([float(rc)] * 128)))
            state.xmem.write_address(96 * 512, mask_data)
        else:
            write_nx8_row(state, 32, [r0] * 128)
            write_nx8_row(state, 64, [rc] * 128)
            state.xmem.write_address(96 * 128, mask_data)
        run_until_complete(state)
        return struct.unpack_from("<128f", state.regfile.get_r_acc_bytes())

    @pytest.mark.parametrize("fmt", [ElementFormat.NX8, ElementFormat.FP32], ids=str)
    def test_mask_affects_multiplication(self, fmt):
        """First 64 bits of mask set -> first 64 MULT_RES lanes active, the rest padded 0."""
        words = self._masked(bytes([0xFF] * 8) + bytes(120), 0, 0, fmt=fmt)
        assert words == (6.0,) * 64 + (0.0,) * 64

    def test_mask_with_shift(self):
        """Mask slot 1, shift_idx=+1 -> 32 bits in slot 1 shift left by 1 (bits 1–32 active)."""
        mask_data = bytes(16) + bytes([0xFF] * 4) + bytes(108)
        words = self._masked(mask_data, 1, 1, r0=5, rc=4)
        assert words[0] == 0
        assert words[1:33] == (20.0,) * 32
        assert words[33:] == (0.0,) * 95

    @pytest.mark.parametrize("fmt", [ElementFormat.NX8, ElementFormat.FP32], ids=str)
    @pytest.mark.parametrize(
        ("mask_type", "pad"), [(MaskType.POS_INF, float("inf")), (MaskType.NEG_INF, float("-inf"))]
    )
    def test_mask_pad_infinities(self, fmt, mask_type, pad):
        """DS mask_type +inf / -inf fills masked-out MULT_RES elements (both modes)."""
        words = self._masked(
            bytes([0xFF] * 8) + bytes(120),
            0,
            0,
            ds_half=ds(128, mask_type=mask_type),
            fmt=fmt,
            r0=0,
            rc=0,
        )
        assert words == (0.0,) * 64 + (pad,) * 64

    def test_mask_type_nan_is_a_configuration_error(self):
        """NX8 has no NaN: a DS whose mask_type is NaN is refused in the check phase."""
        bad = ds(128) | (int(MaskType.NAN) << 10)
        with pytest.raises(CrContentError, match="DS half"):
            self._masked(bytes([0xFF] * 16) + bytes(112), 0, 0, ds_half=bad)


# ============================================================================
# Control Flow
# ============================================================================


class TestControlFlow:
    def test_unconditional_branch(self):
        state = run(
            """\
SET lr0 cr8;;
b skip_section;;
SET lr0 cr9;;
skip_section:
SET lr1 cr10;;
HALT;;
""",
            cr={8: 1, 9: 2, 10: 3},
        )
        assert state.regfile.get_lr(0) == 1
        assert state.regfile.get_lr(1) == 3

    def test_bne(self):
        state = run(
            """\
SET lr0 cr8;;
SET lr1 cr9;;
BNE lr0 lr1 not_equal_branch;;
SET lr2 cr10;;
HALT;;
not_equal_branch:
SET lr2 cr11;;
HALT;;
""",
            cr={8: 10, 9: 20, 10: 0, 11: 1},
        )
        assert state.regfile.get_lr(2) == 1

    def test_beq(self):
        state = run(
            """\
SET lr0 cr8;;
SET lr1 cr8;;
BEQ lr0 lr1 equal_branch;;
SET lr2 cr9;;
HALT;;
equal_branch:
SET lr2 cr10;;
HALT;;
""",
            cr={8: 42, 9: 0, 10: 1},
        )
        assert state.regfile.get_lr(2) == 1

    def test_blt(self):
        state = run(
            """\
SET lr0 cr8;;
SET lr1 cr9;;
BLT.S lr0 lr1 less_branch;;
SET lr2 cr10;;
HALT;;
less_branch:
SET lr2 cr11;;
HALT;;
""",
            cr={8: 5, 9: 6, 10: 0, 11: 1},
        )
        assert state.regfile.get_lr(2) == 1

    def test_bge(self):
        state = run(
            """\
SET lr0 cr8;;
SET lr1 cr9;;
BGE.S lr0 lr1 ge_branch;;
SET lr2 cr10;;
HALT;;
ge_branch:
SET lr2 cr11;;
HALT;;
""",
            cr={8: 6, 9: 5, 10: 0, 11: 1},
        )
        assert state.regfile.get_lr(2) == 1

    def test_bge_not_taken(self):
        state = run(
            """\
SET lr0 cr8;;
SET lr1 cr9;;
BGE.S lr0 lr1 ge_branch_not_taken;;
SET lr2 cr10;;
HALT;;
ge_branch_not_taken:
SET lr2 cr11;;
HALT;;
""",
            cr={8: 5, 9: 6, 10: 1, 11: 0},
        )
        assert state.regfile.get_lr(2) == 1

    def test_bnz(self):
        state = run(
            """\
SET lr0 cr8;;
BNZ lr0 nonzero_branch;;
SET lr2 cr9;;
HALT;;
nonzero_branch:
SET lr2 cr8;;
HALT;;
""",
            cr={8: 1, 9: 0},
        )
        assert state.regfile.get_lr(2) == 1

    def test_bz(self):
        state = run(
            """\
SET lr0 cr8;;
BZ lr0 zero_branch;;
SET lr2 cr8;;
HALT;;
zero_branch:
SET lr2 cr9;;
HALT;;
""",
            cr={8: 0, 9: 1},
        )
        assert state.regfile.get_lr(2) == 1

    def test_bne_lr_cr(self):
        """bne branches when LR does not equal a CR constant."""
        state = make_state(
            """\
SET lr0 cr8;;
BNE lr0 cr1 bne_cr_branch;;
SET lr2 cr9;;
HALT;;
bne_cr_branch:
SET lr2 cr10;;
HALT;;
""",
            cr={8: 10, 9: 0, 10: 1},
        )
        run_until_complete(state)
        assert state.regfile.get_lr(2) == 1

    def test_beq_lr_cr(self):
        """beq branches when LR equals a CR constant."""
        state = make_state(
            """\
SET lr0 cr8;;
BEQ lr0 cr3 beq_cr_branch;;
SET lr2 cr9;;
HALT;;
beq_cr_branch:
SET lr2 cr10;;
HALT;;
""",
            cr={8: 42, 9: 0, 10: 1},
        )
        state.regfile.set_cr(3, 42)
        run_until_complete(state)
        assert state.regfile.get_lr(2) == 1

    def test_blt_lr_cr(self):
        """blt.s branches when LR is less than a CR constant."""
        state = make_state(
            """\
SET lr0 cr8;;
BLT.S lr0 cr2 blt_cr_branch;;
SET lr2 cr9;;
HALT;;
blt_cr_branch:
SET lr2 cr10;;
HALT;;
""",
            cr={8: 5, 9: 0, 10: 1},
        )
        state.regfile.set_cr(2, 10)
        run_until_complete(state)
        assert state.regfile.get_lr(2) == 1

    def test_blt_negative_counter(self):
        """BLT.S sign-extends at 32 bits: -1 < 0 must branch (issue #142)."""
        state = run(
            """\
SET lr0 cr8;;
SET lr1 cr9;;
BLT.S lr0 lr1 neg_branch;;
SET lr2 cr10;;
HALT;;
neg_branch:
SET lr2 cr11;;
HALT;;
""",
            cr={8: -1, 9: 0, 10: 0, 11: 1},
        )
        assert state.regfile.get_lr(2) == 1

    def test_bge_negative_not_taken(self):
        """BGE.S sign-extends at 32 bits: -1 >= 0 must not branch (issue #142)."""
        state = run(
            """\
SET lr0 cr8;;
SET lr1 cr9;;
BGE.S lr0 lr1 neg_ge_branch;;
SET lr2 cr10;;
HALT;;
neg_ge_branch:
SET lr2 cr11;;
HALT;;
""",
            cr={8: -1, 9: 0, 10: 1, 11: 0},
        )
        assert state.regfile.get_lr(2) == 1

    def test_loop_with_cr_limit(self):
        """Loop using bne against a CR constant instead of an LR."""
        state = make_state(
            """\
SET lr0 cr8;;
cr_loop_start:
INC lr0 1;;
BNE lr0 cr5 cr_loop_start;;
HALT;;
""",
            cr={8: 0},
        )
        state.regfile.set_cr(5, 10)
        run_until_complete(state, max_cycles=1000)
        assert state.regfile.get_lr(0) == 10

    def test_simple_loop(self):
        state = run(
            """\
SET lr0 cr8;;
SET lr1 cr9;;
SET lr2 cr8;;
loop_start:
INC lr0 1;;
BNE lr0 lr1 loop_start;;
HALT;;
""",
            cr={8: 0, 9: 10},
            max_cycles=1000,
        )
        assert state.regfile.get_lr(0) == 10

    def test_BKPT_halts(self):
        state = run(
            """\
SET lr0 cr10;;
HALT;;
SET lr0 cr8;;
""",
            cr={8: 0, 9: 10, 10: 99},
        )
        # HALT sets PC = INST_MEM_SIZE, so `SET lr0 0` is never reached
        assert state.regfile.get_lr(0) == 99

    def test_br_cr(self):
        """br accepts a CR register as the branch target address."""
        from ipu_as.session import AssemblySession

        source = """\
BR cr2;;
SET lr0 cr8;;
HALT;;
br_cr_target:
SET lr0 cr9;;
HALT;;
"""
        state = make_state(source, cr={8: 0, 9: 1})
        target_addr = AssemblySession(source).assemble().symbols["br_cr_target"]
        state.regfile.set_cr(2, target_addr)
        run_until_complete(state)
        assert state.regfile.get_lr(0) == 1


# ============================================================================
# Accumulator
# ============================================================================


class TestAccumulator:
    def test_acc_add_first(self):
        """ACC.ADD.FIRST sets r_acc to mult_res without adding previous r_acc."""
        state = make_state(
            """\
ACC.ADD.FIRST;;
HALT;;
"""
        )
        # Pre-fill acc with garbage; ACC.ADD.FIRST should ignore it
        for i in range(128):
            state.regfile.set_r_acc_word(i, 9999)
        mult_buf = state.regfile.mutable("mult_res")
        for i in range(128):
            struct.pack_into("<i", mult_buf, i * 4, 7)
        run_until_complete(state)
        for i in range(128):
            assert state.regfile.get_r_acc_word(i) == 7, (
                f"word {i}: expected 7, got {state.regfile.get_r_acc_word(i)}"
            )

    def test_acc_add_accumulates(self):
        """ACC.ADD adds mult_res to existing r_acc (running sum)."""
        state = make_state(
            """\
ACC.ADD;;
HALT;;
"""
        )
        for i in range(128):
            state.regfile.set_r_acc_word(i, 10)
        mult_buf = state.regfile.mutable("mult_res")
        for i in range(128):
            struct.pack_into("<i", mult_buf, i * 4, 3)
        run_until_complete(state)
        for i in range(128):
            assert state.regfile.get_r_acc_word(i) == 13, (
                f"word {i}: expected 13, got {state.regfile.get_r_acc_word(i)}"
            )

    def test_acc_max_takes_larger(self):
        """ACC.MAX: each element takes max(R_ACC[i], MULT_RES[i])."""
        state = make_state(
            """\
ACC.MAX;;
HALT;;
"""
        )
        for i in range(128):
            # alternate: even lanes R_ACC>MULT_RES, odd lanes R_ACC<MULT_RES
            state.regfile.set_r_acc_word(i, 20 if i % 2 == 0 else 5)
        mult_buf = state.regfile.mutable("mult_res")
        for i in range(128):
            struct.pack_into("<i", mult_buf, i * 4, 10)
        run_until_complete(state)
        for i in range(128):
            expected = 20 if i % 2 == 0 else 10
            got = state.regfile.get_r_acc_word(i)
            assert got == expected, f"word {i}: expected {expected}, got {got}"

    def test_acc_max_first_overwrites(self):
        """ACC.MAX.FIRST: unconditionally overwrites R_ACC with MULT_RES (clean init)."""
        state = make_state(
            """\
ACC.MAX.FIRST;;
HALT;;
"""
        )
        for i in range(128):
            state.regfile.set_r_acc_word(i, 9999)
        mult_buf = state.regfile.mutable("mult_res")
        for i in range(128):
            struct.pack_into("<i", mult_buf, i * 4, 42)
        run_until_complete(state)
        for i in range(128):
            got = state.regfile.get_r_acc_word(i)
            assert got == 42, f"word {i}: expected 42, got {got}"

    def test_acc_sub_subtracts(self):
        """ACC.SUB: subtracts MULT_RES from each R_ACC element (running subtract)."""
        state = make_state(
            """\
ACC.SUB;;
HALT;;
"""
        )
        for i in range(128):
            state.regfile.set_r_acc_word(i, 100)
        mult_buf = state.regfile.mutable("mult_res")
        for i in range(128):
            struct.pack_into("<i", mult_buf, i * 4, 7)
        run_until_complete(state)
        for i in range(128):
            got = state.regfile.get_r_acc_word(i)
            assert got == 93, f"word {i}: expected 93, got {got}"

    def test_acc_sub_first_negates(self):
        """ACC.SUB.FIRST: sets each R_ACC element to negated MULT_RES (clean init)."""
        state = make_state(
            """\
ACC.SUB.FIRST;;
HALT;;
"""
        )
        state.regfile.set_r_acc_bytes(struct.pack("<128f", *([9999.0] * 128)))
        state.regfile.mutable("mult_res")[:] = struct.pack("<128f", *([5.0] * 128))
        run_until_complete(state)
        assert acc_lanes(state) == (-5.0,) * 128

    def test_acc_stride_no_stride(self):
        """ACC.STRIDE with both strides off copies all 128 mult_res words to r_acc from start 0."""
        state = make_state(
            """\
SET lr0 cr8;;
ACC.STRIDE 16 off off lr0;;
HALT;;
""",
            cr={8: 0},
        )
        mult_buf = state.regfile.mutable("mult_res")
        for i in range(128):
            struct.pack_into("<i", mult_buf, i * 4, i)
        run_until_complete(state)
        for i in range(128):
            w = state.regfile.get_r_acc_word(i)
            assert w == i, f"word {i}: expected {i}, got {w}"

    def test_acc_stride_horizontal(self):
        """ACC.STRIDE with horizontal on: take every 2nd column → 64 elements at r_acc[0:64]."""
        state = make_state(
            """\
SET lr0 cr8;;
ACC.STRIDE 16 on off lr0;;
HALT;;
""",
            cr={8: 0},
        )
        mult_buf = state.regfile.mutable("mult_res")
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
        state = make_state(
            """\
SET lr0 cr8;;
ACC.STRIDE 16 on off lr0;;
HALT;;
""",
            cr={8: 1},
        )
        # lr0=1 → offset % 4 = 1 → start index 32. Horizontal on → 64 elements.
        for i in range(128):
            state.regfile.set_r_acc_word(i, 0)
        mult_buf = state.regfile.mutable("mult_res")
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
            assert w == 100 + expected_src, (
                f"word {32 + out_i}: expected {100 + expected_src}, got {w}"
            )
        for i in range(96, 128):
            w = state.regfile.get_r_acc_word(i)
            assert w == 0, f"word {i} (after segment): expected 0, got {w}"

    def test_agg_sum_first_basic(self):
        """AGG.SUM.FIRST: sum all 128 MULT_RES elements and write to R_ACC[dest] (clean init)."""
        state = make_state(
            """\
AGG.SUM.FIRST LR0 cr15.l;;
HALT;;
"""
        )
        state.set_cr_half(15, False, ds(128))
        state.regfile.set_lr(0, 127)  # dest = R_ACC[127]
        for i in range(128):
            state.regfile.set_mult_res_word(i, struct.unpack("<I", struct.pack("<i", 1))[0])
        run_until_complete(state)
        raw = state.regfile.get_r_acc_word(127)
        result = struct.unpack("<i", struct.pack("<I", raw))[0]
        assert result == 128, f"expected sum=128, got {result}"

    def test_agg_sum_first_ignores_existing_dest(self):
        """AGG.SUM.FIRST: existing R_ACC[dest] is NOT added to the result (clean initialisation).

        Dest is placed outside the active element range so its value would be added
        as a seed in the non-FIRST variant, but must be ignored here.
        """
        state = make_state(
            """\
AGG.SUM.FIRST LR0 cr15.l;;
HALT;;
"""
        )
        state.set_cr_half(15, False, ds(4))  # only lanes 0..3 active
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
        state = make_state(
            """\
AGG.SUM.FIRST LR0 cr15.l;;
HALT;;
"""
        )
        state.set_cr_half(15, False, ds(4))
        state.regfile.set_lr(0, 127)
        for i in range(128):
            v = 10 if i < 4 else 1000
            state.regfile.set_mult_res_word(i, struct.unpack("<I", struct.pack("<i", v))[0])
        run_until_complete(state)
        raw = state.regfile.get_r_acc_word(127)
        result = struct.unpack("<i", struct.pack("<I", raw))[0]
        assert result == 40, f"expected sum of 4 tens = 40, got {result}"

    def test_agg_sum_first_uses_named_cr_register(self):
        """AGG.SUM.FIRST naming CR3 uses CR3.valid_elements=128, ignoring CR15=4."""
        state = make_state(
            """\
AGG.SUM.FIRST LR0 cr3.l;;
HALT;;
"""
        )
        state.set_cr_half(15, False, ds(4))
        state.regfile.set_cr(3, ds(128))
        state.regfile.set_lr(0, 127)
        for i in range(128):
            state.regfile.set_mult_res_word(i, struct.unpack("<I", struct.pack("<i", 1))[0])
        run_until_complete(state)
        raw = state.regfile.get_r_acc_word(127)
        result = struct.unpack("<i", struct.pack("<I", raw))[0]
        assert result == 128, f"expected sum of all 128 lanes = 128, got {result}"

    def test_agg_sum_accumulates(self):
        """AGG.SUM: sum of active MULT_RES elements is ADDED to existing R_ACC[dest] (running acc).

        Dest is placed outside the active element range so it acts as a pure accumulator
        that is not double-counted in the sum.
        """
        state = make_state(
            """\
AGG.SUM LR0 cr15.l;;
HALT;;
"""
        )
        state.set_cr_half(15, False, ds(64))  # only lanes 0..63 active
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
        """AGG.MAX.FIRST: max of all 128 MULT_RES elements, no seed from dest."""
        state = make_state(
            """\
AGG.MAX.FIRST LR0 cr15.l;;
HALT;;
"""
        )
        state.set_cr_half(15, False, ds(128))
        state.regfile.set_lr(0, 127)
        for i in range(128):
            state.regfile.set_mult_res_word(
                i, struct.unpack("<I", struct.pack("<i", 5 + (i % 10)))[0]
            )
        state.regfile.set_mult_res_word(127, struct.unpack("<I", struct.pack("<i", 9999))[0])
        run_until_complete(state)
        raw = state.regfile.get_r_acc_word(127)
        result = struct.unpack("<i", struct.pack("<I", raw))[0]
        assert result == 9999, f"expected max=9999 (from MULT_RES slot 127), got {result}"

    def test_agg_max_first_ignores_garbage_dest(self):
        """AGG.MAX.FIRST: existing R_ACC[dest] is NOT used as a seed.

        Dest is placed outside the active element range and pre-loaded with a
        large garbage value; a seeded implementation would return it, the
        clean-init implementation must return the max of the active MULT_RES elements.
        """
        state = make_state(
            """\
AGG.MAX.FIRST LR0 cr15.l;;
HALT;;
"""
        )
        state.set_cr_half(15, False, ds(64))  # only lanes 0..63 active
        state.regfile.set_lr(0, 127)  # dest = R_ACC[127], outside active range
        for i in range(128):
            state.regfile.set_mult_res_word(i, struct.unpack("<I", struct.pack("<i", 5))[0])
        state.regfile.set_r_acc_word(127, struct.unpack("<I", struct.pack("<i", 9999))[0])
        run_until_complete(state)
        raw = state.regfile.get_r_acc_word(127)
        result = struct.unpack("<i", struct.pack("<I", raw))[0]
        assert result == 5, (
            f"expected max of active MULT_RES lanes = 5 (not R_ACC seed 9999), got {result}"
        )

    def test_agg_max_first_zero_active_writes_identity_seed(self):
        """AGG.MAX.FIRST with valid_elements=0: dest gets the identity seed (-inf)."""
        state = make_state(
            """\
AGG.MAX.FIRST LR0 cr15.l;;
HALT;;
"""
        )
        state.set_cr_half(15, False, ds(0))  # no active lanes
        state.regfile.set_lr(0, 7)
        state.regfile.set_r_acc_word(7, struct.unpack("<I", struct.pack("<i", 1234))[0])
        run_until_complete(state)
        assert acc_lanes(state)[7] == float("-inf")

    def test_agg_max_first_masks_tail(self):
        """AGG.MAX.FIRST: MULT_RES tail elements beyond valid_elements are excluded."""
        state = make_state(
            """\
AGG.MAX.FIRST LR0 cr15.l;;
HALT;;
"""
        )
        state.set_cr_half(15, False, ds(50))
        state.regfile.set_lr(0, 127)
        for i in range(128):
            v = 3 if i < 50 else 9999
            state.regfile.set_mult_res_word(i, struct.unpack("<I", struct.pack("<i", v))[0])
        run_until_complete(state)
        raw = state.regfile.get_r_acc_word(127)
        result = struct.unpack("<i", struct.pack("<I", raw))[0]
        assert result == 3, f"expected max of active MULT_RES prefix = 3, got {result}"

    def test_agg_max_seed_wins(self):
        """AGG.MAX: existing R_ACC[dest] seed beats all active MULT_RES elements — dest unchanged."""
        state = make_state(
            """\
AGG.MAX LR0 cr15.l;;
HALT;;
"""
        )
        state.set_cr_half(15, False, ds(128))
        state.regfile.set_lr(0, 127)
        for i in range(128):
            state.regfile.set_mult_res_word(i, struct.unpack("<I", struct.pack("<i", 10))[0])
        state.regfile.set_r_acc_word(127, struct.unpack("<I", struct.pack("<i", 99))[0])
        run_until_complete(state)
        raw = state.regfile.get_r_acc_word(127)
        result = struct.unpack("<i", struct.pack("<I", raw))[0]
        assert result == 99, (
            f"expected seed 99 to remain (beats all MULT_RES lanes=10), got {result}"
        )

    def test_agg_max_lane_wins(self):
        """AGG.MAX: an active MULT_RES element beats the existing R_ACC[dest] seed — dest updated."""
        state = make_state(
            """\
AGG.MAX LR0 cr15.l;;
HALT;;
"""
        )
        state.set_cr_half(15, False, ds(100))
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

    def test_agg_and_activate_quantize_same_cycle_sees_live_r_acc(self):
        """AGG.SUM.FIRST in ACC slot and ACTIVATE.QUANTIZE in AAQ slot issued together.

        AGG reads from MULT_RES (live) and writes r_acc[dest]. Since the ACC
        slot dispatches before the AAQ slot within the same VLIW cycle,
        ACTIVATE.QUANTIZE must see that same-cycle write to r_acc, not the
        cycle-start snapshot.
        """
        state = make_state(
            """\
AGG.SUM.FIRST LR0 cr15.l; ACTIVATE.QUANTIZE relu cr15.l cr14.l cr2 lr0;;
HALT;;
""",
            cr={2: 40},
        )
        state.regfile.mutable("mult_res")[:] = struct.pack("<128f", *([3.0] * 128))
        state.regfile.set_r_acc_bytes(struct.pack("<128f", *([3.0] * 128)))
        run_until_complete(state)

        assert acc_lanes(state)[0] == 384.0  # AGG summed this word's MULT_RES (128 x 3)
        # ACTIVATE.QUANTIZE saw the post-AGG lane 0 (384), not the cycle-start 3:
        # 384 sets the row's scale (2^8) and decodes back exactly.
        scale, dynamic = split_metadata(state.xmem.read_metadata(40 * 128))
        assert scale == 127 + 8
        stored = decode_row(bytes(state.xmem.read_address(40 * 128, 128)), E4, scale, dynamic)
        assert stored[0] == 384.0


def _lrd_bytes(*vals: int) -> tuple[int, int]:
    """Pack 8 byte values into (lo_cr, hi_cr) 32-bit little-endian words for an LRDn pair."""
    assert len(vals) == 8
    lo = int.from_bytes(bytes(vals[0:4]), "little")
    hi = int.from_bytes(bytes(vals[4:8]), "little")
    return lo, hi


# ============================================================================


class TestReshape:
    """ACC.RESHAPE: scatter MULT_RES elements into R_ACC via two LrdIdx byte-index arrays (issue #192, #210)."""

    def test_reshape_mask_zero_copies_all_eight_slots(self):
        """reshape_mask=0: all 8 element slots participate; reads from MULT_RES not R_ACC."""
        src_lo, src_hi = _lrd_bytes(0, 1, 2, 3, 4, 5, 6, 7)
        dst_lo, dst_hi = _lrd_bytes(100, 101, 102, 103, 104, 105, 106, 107)
        state = make_state(
            """\
SET lr0 cr8;;
SET lr1 cr9;;
SET lr2 cr10;;
SET lr3 cr11;;
ACC.RESHAPE lrd0 lrd2 0;;
HALT;;
""",
            cr={8: src_lo, 9: src_hi, 10: dst_lo, 11: dst_hi},
        )
        for i in range(8):
            state.regfile.set_mult_res_word(i, 1000 + i)
        for i in range(100, 108):
            state.regfile.set_r_acc_word(i, 9999)
        run_until_complete(state)
        for i in range(8):
            assert state.regfile.get_r_acc_word(100 + i) == 1000 + i

    def test_reshape_mask_seven_copies_only_slot_seven(self):
        """reshape_mask=7: only the trailing element (slot 7) participates; slots 0-6 untouched."""
        src_lo, src_hi = _lrd_bytes(0, 1, 2, 3, 4, 5, 6, 7)
        dst_lo, dst_hi = _lrd_bytes(100, 101, 102, 103, 104, 105, 106, 107)
        state = make_state(
            """\
SET lr0 cr8;;
SET lr1 cr9;;
SET lr2 cr10;;
SET lr3 cr11;;
ACC.RESHAPE lrd0 lrd2 7;;
HALT;;
""",
            cr={8: src_lo, 9: src_hi, 10: dst_lo, 11: dst_hi},
        )
        for i in range(8):
            state.regfile.set_mult_res_word(i, 1000 + i)
        for i in range(100, 108):
            state.regfile.set_r_acc_word(i, 9999)
        run_until_complete(state)
        # Only slot 7 participates: MULT_RES[7]=1007 → R_ACC[107].
        assert state.regfile.get_r_acc_word(107) == 1007
        for i in range(100, 107):
            assert state.regfile.get_r_acc_word(i) == 9999

    def test_reshape_mask_six_copies_trailing_two_slots(self):
        """reshape_mask=6: slots 6 and 7 participate; slots 0-5 are untouched."""
        src_lo, src_hi = _lrd_bytes(10, 11, 12, 13, 14, 15, 20, 21)
        dst_lo, dst_hi = _lrd_bytes(50, 51, 52, 53, 54, 55, 60, 61)
        state = make_state(
            """\
SET lr0 cr8;;
SET lr1 cr9;;
SET lr2 cr10;;
SET lr3 cr11;;
ACC.RESHAPE lrd0 lrd2 6;;
HALT;;
""",
            cr={8: src_lo, 9: src_hi, 10: dst_lo, 11: dst_hi},
        )
        for i in range(128):
            state.regfile.set_mult_res_word(i, 5000 + i)
        for i in range(128):
            state.regfile.set_r_acc_word(i, 9999)
        run_until_complete(state)
        # Slots 6: MULT_RES[20] → R_ACC[60]; slot 7: MULT_RES[21] → R_ACC[61].
        assert state.regfile.get_r_acc_word(60) == 5020
        assert state.regfile.get_r_acc_word(61) == 5021
        # Other dest indices untouched.
        assert state.regfile.get_r_acc_word(50) == 9999
        assert state.regfile.get_r_acc_word(55) == 9999

    def test_reshape_mask_three_masks_lowest_three_slots(self):
        """reshape_mask=3: slots 0-2 (lowest-indexed) are masked out; slots 3-7 participate."""
        src_lo, src_hi = _lrd_bytes(0, 1, 2, 3, 4, 5, 6, 7)
        dst_lo, dst_hi = _lrd_bytes(100, 101, 102, 103, 104, 105, 106, 107)
        state = make_state(
            """\
SET lr0 cr8;;
SET lr1 cr9;;
SET lr2 cr10;;
SET lr3 cr11;;
ACC.RESHAPE lrd0 lrd2 3;;
HALT;;
""",
            cr={8: src_lo, 9: src_hi, 10: dst_lo, 11: dst_hi},
        )
        for i in range(8):
            state.regfile.set_mult_res_word(i, 6000 + i)
        for i in range(100, 108):
            state.regfile.set_r_acc_word(i, 9999)
        run_until_complete(state)
        # Slots 0-2 masked out: dest untouched.
        for i in range(100, 103):
            assert state.regfile.get_r_acc_word(i) == 9999
        # Slots 3-7 participate: MULT_RES[3..7] -> R_ACC[103..107].
        for i in range(3, 8):
            assert state.regfile.get_r_acc_word(100 + i) == 6000 + i

    def test_reshape_raises_on_out_of_range_source(self):
        """A participating source[i] >= 128 raises rather than silently skipping."""
        src_lo, src_hi = _lrd_bytes(200, 1, 2, 3, 4, 5, 6, 7)
        dst_lo, dst_hi = _lrd_bytes(100, 101, 102, 103, 104, 105, 106, 107)
        state = make_state(
            """\
SET lr0 cr8;;
SET lr1 cr9;;
SET lr2 cr10;;
SET lr3 cr11;;
ACC.RESHAPE lrd0 lrd2 0;;
HALT;;
""",
            cr={8: src_lo, 9: src_hi, 10: dst_lo, 11: dst_hi},
        )
        for i in range(8):
            state.regfile.set_mult_res_word(i, 1000 + i)
        for i in range(100, 108):
            state.regfile.set_r_acc_word(i, 9999)
        with pytest.raises(EmulatorFault, match="RESHAPE element") as caught:
            run_until_complete(state)
        assert state.halt == HaltRecord("fault", caught.value.pc, caught.value.fault)
        assert caught.value.fault is FaultId.F5

    def test_reshape_raises_on_out_of_range_dest(self):
        """A participating dest[i] >= 128 raises rather than silently skipping."""
        src_lo, src_hi = _lrd_bytes(0, 1, 2, 3, 4, 5, 6, 7)
        dst_lo, dst_hi = _lrd_bytes(200, 101, 102, 103, 104, 105, 106, 107)
        state = make_state(
            """\
SET lr0 cr8;;
SET lr1 cr9;;
SET lr2 cr10;;
SET lr3 cr11;;
ACC.RESHAPE lrd0 lrd2 0;;
HALT;;
""",
            cr={8: src_lo, 9: src_hi, 10: dst_lo, 11: dst_hi},
        )
        for i in range(8):
            state.regfile.set_mult_res_word(i, 1000 + i)
        for i in range(100, 108):
            state.regfile.set_r_acc_word(i, 9999)
        with pytest.raises(EmulatorFault, match="RESHAPE element") as caught:
            run_until_complete(state)
        assert state.halt == HaltRecord("fault", caught.value.pc, caught.value.fault)
        assert caught.value.fault is FaultId.F5

    def test_reshape_non_participating_out_of_range_does_not_raise(self):
        """An out-of-range source/dest outside the participation range (masked out) is fine."""
        src_lo, src_hi = _lrd_bytes(200, 1, 2, 3, 4, 5, 6, 7)
        dst_lo, dst_hi = _lrd_bytes(200, 101, 102, 103, 104, 105, 106, 107)
        state = make_state(
            """\
SET lr0 cr8;;
SET lr1 cr9;;
SET lr2 cr10;;
SET lr3 cr11;;
ACC.RESHAPE lrd0 lrd2 1;;
HALT;;
""",
            cr={8: src_lo, 9: src_hi, 10: dst_lo, 11: dst_hi},
        )
        for i in range(8):
            state.regfile.set_mult_res_word(i, 1000 + i)
        for i in range(100, 108):
            state.regfile.set_r_acc_word(i, 9999)
        run_until_complete(state)
        # slot 0 (out-of-range) does not participate under mask=1; slots 1-7 do.
        for i in range(1, 8):
            assert state.regfile.get_r_acc_word(100 + i) == 1000 + i

    def test_reshape_reads_from_mult_res_not_r_acc(self):
        """ACC.RESHAPE copies from MULT_RES, not R_ACC; R_ACC source values are ignored."""
        src_lo, src_hi = _lrd_bytes(0, 1, 2, 3, 4, 5, 6, 7)
        dst_lo, dst_hi = _lrd_bytes(10, 11, 12, 13, 14, 15, 16, 17)
        state = make_state(
            """\
SET lr0 cr8;;
SET lr1 cr9;;
SET lr2 cr10;;
SET lr3 cr11;;
ACC.RESHAPE lrd0 lrd2 0;;
HALT;;
""",
            cr={8: src_lo, 9: src_hi, 10: dst_lo, 11: dst_hi},
        )
        for i in range(8):
            state.regfile.set_mult_res_word(i, 2000 + i)
            state.regfile.set_r_acc_word(i, 9999)  # must NOT be read by ACC.RESHAPE
        for i in range(10, 18):
            state.regfile.set_r_acc_word(i, 0)
        run_until_complete(state)
        for i in range(8):
            assert state.regfile.get_r_acc_word(10 + i) == 2000 + i, (
                f"R_ACC[{10 + i}] should be MULT_RES[{i}]=={2000 + i}, not R_ACC[{i}]"
            )

    def test_reshape_mask_lr_mode_uses_lr_low_bits(self):
        """reshape_mask=LR4 with LR4.value=6: only slots 6 and 7 participate.

        LR0-LR3 are used by the LRD pair setup (SET instructions), so LR4
        is used as the mask register to avoid aliasing.
        """
        src_lo, src_hi = _lrd_bytes(0, 1, 2, 3, 4, 5, 6, 7)
        dst_lo, dst_hi = _lrd_bytes(100, 101, 102, 103, 104, 105, 106, 107)
        state = make_state(
            """\
SET lr4 cr12;;
SET lr0 cr8;;
SET lr1 cr9;;
SET lr2 cr10;;
SET lr3 cr11;;
ACC.RESHAPE lrd0 lrd2 lr4;;
HALT;;
""",
            cr={8: src_lo, 9: src_hi, 10: dst_lo, 11: dst_hi, 12: 6},
        )
        for i in range(8):
            state.regfile.set_mult_res_word(i, 3000 + i)
        for i in range(100, 108):
            state.regfile.set_r_acc_word(i, 9999)
        run_until_complete(state)
        assert state.regfile.get_r_acc_word(106) == 3006
        assert state.regfile.get_r_acc_word(107) == 3007
        for i in range(100, 106):
            assert state.regfile.get_r_acc_word(i) == 9999

    def test_reshape_mask_lr_value_eight_participates_in_none(self):
        """LR4 value 8 (== RESHAPE_ELEMENT_COUNT) is valid: zero elements participate."""
        src_lo, src_hi = _lrd_bytes(0, 1, 2, 3, 4, 5, 6, 7)
        dst_lo, dst_hi = _lrd_bytes(100, 101, 102, 103, 104, 105, 106, 107)
        state = make_state(
            """\
SET lr4 cr12;;
SET lr0 cr8;;
SET lr1 cr9;;
SET lr2 cr10;;
SET lr3 cr11;;
ACC.RESHAPE lrd0 lrd2 lr4;;
HALT;;
""",
            cr={8: src_lo, 9: src_hi, 10: dst_lo, 11: dst_hi, 12: 8},
        )
        for i in range(8):
            state.regfile.set_mult_res_word(i, 4000 + i)
        for i in range(100, 108):
            state.regfile.set_r_acc_word(i, 9999)
        run_until_complete(state)
        for i in range(100, 108):
            assert state.regfile.get_r_acc_word(i) == 9999

    def test_reshape_mask_lr_value_exceeding_element_count_raises(self):
        """LR4 value 14 exceeds RESHAPE_ELEMENT_COUNT (8): must raise, not clamp."""
        src_lo, src_hi = _lrd_bytes(0, 1, 2, 3, 4, 5, 6, 7)
        dst_lo, dst_hi = _lrd_bytes(100, 101, 102, 103, 104, 105, 106, 107)
        state = make_state(
            """\
SET lr4 cr12;;
SET lr0 cr8;;
SET lr1 cr9;;
SET lr2 cr10;;
SET lr3 cr11;;
ACC.RESHAPE lrd0 lrd2 lr4;;
HALT;;
""",
            cr={8: src_lo, 9: src_hi, 10: dst_lo, 11: dst_hi, 12: 14},
        )
        for i in range(8):
            state.regfile.set_mult_res_word(i, 4000 + i)
        for i in range(100, 108):
            state.regfile.set_r_acc_word(i, 9999)
        with pytest.raises(EmulatorFault, match="reshape_mask") as caught:
            run_until_complete(state)
        assert state.halt == HaltRecord("fault", caught.value.pc, caught.value.fault)
        assert caught.value.fault is FaultId.F5

    def test_reshape_reads_same_cycle_mult_res(self):
        """A MULT in the same word is seen: RESHAPE scatters this word's product
        (Q32 rule 3, E04-26), not the MULT_RES the cycle started with."""
        src_lo, src_hi = _lrd_bytes(0, 1, 2, 3, 4, 5, 6, 7)
        dst_lo, dst_hi = _lrd_bytes(10, 11, 12, 13, 14, 15, 16, 17)
        state = make_state(
            """\
SET lr0 cr8;;
SET lr1 cr9;;
SET lr2 cr10;;
SET lr3 cr11;;
MULT.RC.VS lr4 0 lr4 cr15.l cr14.l; ACC.RESHAPE lrd0 lrd2 6;;
HALT;;
""",
            cr={8: src_lo, 9: src_hi, 10: dst_lo, 11: dst_hi},
        )
        # R_CYCLIC[i] = i + 1, so this word's MULT_RES[i] = (i + 1) ** 2.
        fill_nx8_cyclic(state, list(range(1, 9)))
        # The MULT_RES the cycle starts with; RESHAPE must not see it.
        state.regfile.mutable("mult_res")[:32] = struct.pack("<8f", *[700.0 + i for i in range(8)])
        state.regfile.set_r_acc_bytes(struct.pack("<128f", *([9999.0] * 128)))
        run_until_complete(state)
        # Slots 6 and 7: src[6]=6 -> dest[6]=16; src[7]=7 -> dest[7]=17.
        lanes = acc_lanes(state)
        assert (lanes[16], lanes[17], lanes[15]) == (49.0, 64.0, 9999.0)


def _step(state: IpuState) -> CycleOutcome:
    """One cycle on *state*, honouring BREAK (``Cycle.step``)."""
    ipu = Ipu(state)
    return ipu.cycle.step()


class TestProgramCounter:
    def test_pc_advances(self):
        state = make_state(
            """\
SET lr0 cr8;;
SET lr1 cr9;;
HALT;;
""",
            cr={8: 100, 9: 200},
        )
        assert state.program_counter == 0
        _step(state)
        assert state.program_counter == 1
        _step(state)
        assert state.program_counter == 2


# ============================================================================
# Breakpoints
# ============================================================================


class TestBreakpoints:
    def test_break_stops_execution(self):
        state = make_state(
            """\
SET lr0 cr8;;
BREAK;;
SET lr0 cr9;;
HALT;;
""",
            cr={8: 1, 9: 2},
        )
        r = _step(state)
        assert r == CycleOutcome.RETIRED
        assert state.regfile.get_lr(0) == 1

        r = _step(state)
        assert r == CycleOutcome.BREAK

    def test_break_ifeq(self):
        state = make_state(
            """\
SET lr5 cr8;;
BREAK.IFEQ lr5 42;;
HALT;;
""",
            cr={8: 42},
        )
        _step(state)  # SET lr5 42
        assert state.regfile.get_lr(5) == 42

        r = _step(state)
        assert r == CycleOutcome.BREAK

    def test_break_ifeq_no_match(self):
        state = make_state(
            """\
SET lr5 cr8;;
BREAK.IFEQ lr5 42;;
HALT;;
""",
            cr={8: 10},
        )
        _step(state)
        r = _step(state)
        assert r == CycleOutcome.RETIRED

    def test_run_with_debug_step(self):
        actions = iter([DebugAction.STEP, DebugAction.CONTINUE])

        def callback(state, cycle):
            return next(actions, DebugAction.CONTINUE)

        state = make_state(
            """\
BREAK;;
SET lr0 cr8;;
HALT;;
""",
            cr={8: 1},
        )
        run_with_debug(state, callback)
        assert state.regfile.get_lr(0) == 1


# ============================================================================
# Decode / encode round-trip
# ============================================================================


class TestDebuggerStops:
    def test_step_stops_before_next_instruction_and_counts_completed_cycles(self):
        state = make_state("BREAK;;\nINC lr0 1;;\nINC lr0 1;;\nHALT;;")
        stops = []

        def callback(state, cycle):
            stops.append((state.program_counter, cycle, state.regfile.get_lr(0)))
            return DebugAction.STEP

        assert run_with_debug(state, callback) == 4
        assert stops == [(0, 0, 0), (1, 1, 0), (2, 2, 1), (3, 3, 2)]
        assert state.stats.total_cycles == 4

    def test_runtime_breakpoint_rearms_on_loop_visit(self):
        state = make_state("loop:\nINC lr0 1;;\nBNE lr0 cr2 loop;;\nHALT;;", cr={2: 3})
        control = BreakpointRunControl()
        control.add_breakpoint(0)
        original = [inst.copy() if inst else None for inst in state.inst_mem]
        stops = []

        def callback(state, cycle):
            stops.append((state.program_counter, cycle, state.regfile.get_lr(0)))
            assert control.stop_reason == "breakpoint"
            return DebugAction.CONTINUE

        assert run_with_debug(state, callback, control=control) == 7
        assert stops == [(0, 0, 0), (0, 2, 1), (0, 4, 2)]
        assert state.inst_mem == original

    def test_simultaneous_causes_stop_once(self):
        state = make_state("BREAK;;\nBREAK;;\nHALT;;")
        control = BreakpointRunControl()
        control.add_breakpoint(1)
        reasons = []

        def callback(state, cycle):
            reasons.append((state.program_counter, cycle, control.stop_reason))
            if cycle == 0:
                control.run_until(1, state.program_counter)
                return DebugAction.STEP
            return DebugAction.CONTINUE

        assert run_with_debug(state, callback, control=control) == 3
        assert reasons == [
            (0, 0, "BREAK instruction"),
            (1, 1, "breakpoint, run target, step, BREAK instruction"),
        ]
        assert control.run_target is None

    def test_run_target_stops_before_side_effects(self):
        state = make_state("INC lr0 1;;\nINC lr0 1;;\nHALT;;")
        control = BreakpointRunControl()
        control.run_until(1, 0)
        stops = []

        def callback(state, cycle):
            stops.append((state.program_counter, cycle, state.regfile.get_lr(0)))
            assert control.run_target is None
            assert control.stop_reason == "run target"
            return DebugAction.QUIT

        assert run_with_debug(state, callback, control=control) == 1
        assert stops == [(1, 1, 1)]
        assert state.stats.total_cycles == 1

    def test_intervening_break_cancels_run_target(self):
        state = make_state("BREAK;;\nBREAK;;\nINC lr0 1;;\nHALT;;")
        control = BreakpointRunControl()
        stops = []

        def callback(state, cycle):
            stops.append(state.program_counter)
            if cycle == 0:
                control.run_until(2, state.program_counter)
            else:
                assert control.run_target is None
            return DebugAction.CONTINUE

        assert run_with_debug(state, callback, control=control) == 4
        assert stops == [0, 1]
        assert state.regfile.get_lr(0) == 1

    @pytest.mark.parametrize("termination", ["halt", "quit", "limit", "error"])
    def test_run_target_cleared_on_every_exit(self, termination):
        state = make_state("BREAK;;\nHALT;;")
        control = BreakpointRunControl()

        def callback(state, cycle):
            control.run_until(10, state.program_counter)
            if termination == "error":
                raise ValueError("callback failed")
            return DebugAction.QUIT if termination == "quit" else DebugAction.CONTINUE

        if termination in ("limit", "error"):
            with pytest.raises(EmulatorFault if termination == "limit" else ValueError):
                run_with_debug(state, callback, max_cycles=1, control=control)
        else:
            run_with_debug(state, callback, control=control)
        assert control.run_target is None

    def test_importing_the_emulator_imports_no_debugger(self):
        # A fresh interpreter, so this test file's own imports do not count.
        probe = (
            "import sys, ipu_emu; "
            "print(sorted(m for m in sys.modules if m.startswith(('ipu_emu.debug', 'ipu_debug'))))"
        )
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
        result = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, env=env, check=True
        )
        assert result.stdout.strip() == "[]"

    def test_two_states_keep_their_own_controls(self):
        program = "INC lr0 1;;\nINC lr0 1;;\nINC lr0 1;;\nHALT;;"
        first, second = make_state(program), make_state(program)
        first_control, second_control = BreakpointRunControl(), BreakpointRunControl()
        first_control.add_breakpoint(1)
        second_control.add_breakpoint(2)
        stops = []

        def on_second(state, cycle):
            stops.append(("second", state.program_counter))
            return DebugAction.CONTINUE

        def on_first(state, cycle):
            stops.append(("first", state.program_counter))
            # The second runs to completion while the first is paused.
            run_with_debug(second, on_second, control=second_control)
            return DebugAction.CONTINUE

        run_with_debug(first, on_first, control=first_control)
        assert stops == [("first", 1), ("second", 2)]
        assert first_control.breakpoints == {1} and second_control.breakpoints == {2}
        assert (first_control.last_completed_pc, second_control.last_completed_pc) == (3, 3)

    def test_without_a_control_only_embedded_breaks_stop(self):
        state = make_state("INC lr0 1;;\nBREAK;;\nHALT;;")
        stops = []

        def callback(state, cycle):
            stops.append(state.program_counter)
            return DebugAction.CONTINUE

        assert run_with_debug(state, callback) == 3
        assert stops == [1]

    def test_pc_and_register_edits_apply_to_resumed_instruction(self):
        state = make_state("BREAK;;\nSET lr1 cr2;;\nHALT;;")

        def callback(state, cycle):
            state.program_counter = 1
            state.regfile.set_cr(2, 19)
            return DebugAction.STEP

        stops = []

        def edit_once(state, cycle):
            stops.append((state.program_counter, cycle, state.regfile.get_lr(1)))
            return callback(state, cycle) if cycle == 0 else DebugAction.QUIT

        assert run_with_debug(state, edit_once) == 1
        assert stops == [(0, 0, 0), (2, 1, 19)]


class TestDecodeRoundtrip:
    def test_nop_decodes(self):
        """An all-NOP instruction should decode without error."""
        fields = DecodedWord.from_word(0)
        assert isinstance(fields, dict)
        assert len(fields) > 0

    def test_roundtrip(self):
        """Assemble → decode → verify known fields."""
        encoded = assemble("SET lr13 cr8;;\nHALT;;")
        assert len(encoded) == 2
        d = DecodedWord.from_word(encoded[0])
        # LR opcode should be 'set' = index 0
        assert d["lr_inst_0_token_0_lr_inst_opcode"] == 0  # set
        assert d["lr_inst_0_token_2_lr_reg_field"] == 13  # reg = lr13
        assert (
            d["lr_inst_0_token_1_addbi_immediate"] == 16 + 8
        )  # src = cr8 (LcrHalfIdx: CRs follow the 16 LRs)


# ============================================================================
# FP8 multiply
# ============================================================================


class TestNx8Multiply:
    @pytest.mark.parametrize(
        "fmt",
        [E4, Nx8Format(sign=1, exp_format=5, num_bits=8), Nx8Format(1, 0, 8, dynamic=True)],
        ids=["e4", "e5", "dynamic"],
    )
    def test_nx8_mult_rc_vv(self, fmt):
        """NX8 rows quantized in format *fmt*: 1.0 x 2.0 -> 2.0 for every element."""

        state = make_state(
            """\
SET lr0 cr8;;
LD.MULT r0 lr0 cr0;;
SET lr1 cr9;;
SET lr2 cr10;;
LD.MULT.CYCLIC lr1 cr0 lr2;;
SET lr5 cr10;;
SET lr6 cr10;;
MULT.RC.VV lr6 r0 0 lr5 cr15.l cr14.l;
ACC.ADD;;
HALT;;
""",
            cr={8: 32, 9: 64, 10: 0},
            halves={"cr14.l": qf(fmt)},
        )
        write_nx8_row(state, 32, [1.0] * 128, fmt)
        write_nx8_row(state, 64, [2.0] * 128, fmt)
        run_until_complete(state)

        assert acc_lanes(state) == (2.0,) * 128


# ============================================================================
# MULT.RC.VE
# ============================================================================


class TestMultRcVe:
    """Tests for the MULT.RC.VE instruction."""

    # ------------------------------------------------------------------
    # MULT.RC.VE with a CR-encoded scalar source
    # ------------------------------------------------------------------

    def test_mult_rc_ve_cr_constant(self):
        """MULT.RC.VE: a CR constant (3.0) x RC elements (2.0) -> 6.0."""

        state = make_state(
            """\
SET lr0 cr8;;
SET lr1 cr9;;
LD.MULT.CYCLIC lr0 cr0 lr1;;
SET lr2 cr9;;
SET lr4 cr9;;
MULT.RC.VE lr2 cr3 0 lr4 cr15.l cr14.l;
ACC.ADD;;
HALT;;
""",
            cr={8: 32, 9: 0},
        )
        state.regfile.set_cr(3, int_constant(3))  # scalar 3.0, a MULT constant
        write_nx8_row(state, 32, [2] * 128)
        run_until_complete(state)

        assert acc_lanes(state) == (6.0,) * 128

    def test_mult_rc_ve_cr_negative_constant(self):
        """MULT.RC.VE: a negative constant (-2.0) x positive RC elements (5.0) -> -10.0."""

        state = make_state(
            """\
SET lr0 cr8;;
SET lr1 cr9;;
LD.MULT.CYCLIC lr0 cr0 lr1;;
SET lr2 cr9;;
SET lr4 cr9;;
MULT.RC.VE lr2 cr2 0 lr4 cr15.l cr14.l;
ACC.ADD;;
HALT;;
""",
            cr={8: 32, 9: 0},
        )
        state.regfile.set_cr(2, int_constant(-2))
        write_nx8_row(state, 32, [5] * 128)
        run_until_complete(state)

        assert acc_lanes(state) == (-10.0,) * 128

    def test_mult_rc_ve_cr_wraps_at_rc_boundary(self):
        """MULT.RC.VE: RC indices wrap modulo 512 (cyclic, no padding)."""
        # rc_idx = 450, so bytes 450..511 then 0..65 of RC are used — all rc_fill.
        rc_fill = 4
        scalar = 7

        state = make_state(
            """\
SET lr2 cr8;;
SET lr4 cr9;;
MULT.RC.VE lr2 cr3 0 lr4 cr15.l cr14.l;
ACC.ADD;;
HALT;;
""",
            cr={8: 450, 9: 0},
        )
        state.regfile.set_cr(3, int_constant(scalar))
        fill_nx8_cyclic(state, [rc_fill] * 512)
        run_until_complete(state)

        assert acc_lanes(state) == (float(scalar * rc_fill),) * 128

    def test_mult_rc_ve_cr_wraps_with_per_slot_metadata(self):
        """MULT.RC.VE: the window wraps modulo 512 across slots whose metadata differ."""
        state = make_state(
            """\
SET lr2 cr8;;
SET lr4 cr9;;
MULT.RC.VE lr2 cr6 0 lr4 cr15.l cr14.l;
ACC.ADD;;
HALT;;
""",
            cr={8: 500, 9: 0},
        )
        state.regfile.set_cr(6, int_constant(3.0))
        # Slot 3 holds 2.0 (one scale), slot 0 holds 0.25 (another scale).
        fill_nx8_cyclic(state, [0.25] * 128 + [1.0] * 256 + [2.0] * 128)
        run_until_complete(state)

        want = (6.0,) * 12 + (0.75,) * 116  # elements 500..511, then 0..115
        assert acc_lanes(state) == want

    # ------------------------------------------------------------------
    # MULT.RC.VV / MULT.RC.VE behave correctly alongside each other
    # ------------------------------------------------------------------

    def test_mult_rc_vv_still_works(self):
        """MULT.RC.VV still works correctly after adding MULT.RC.VE."""
        r0_data = [4] * 128
        cyclic_data = [5] * 128

        state = make_state(
            """\
SET lr0 cr8;;
LD.MULT r0 lr0 cr0;;
SET lr1 cr9;;
SET lr2 cr10;;
LD.MULT.CYCLIC lr1 cr0 lr2;;
MULT.RC.VV lr2 r0 0 lr2 cr15.l cr14.l;
ACC.ADD;;
HALT;;
""",
            cr={8: 32, 9: 64, 10: 0},
        )
        write_nx8_row(state, 0x1000 // 128, r0_data)
        write_nx8_row(state, 0x2000 // 128, cyclic_data)
        run_until_complete(state)

        acc_raw = state.regfile.mutable("r_acc")
        for i in range(128):
            val = struct.unpack_from("<f", acc_raw, i * 4)[0]
            assert val == 20, f"acc word {i}: expected 20, got {val}"

    def test_mult_rc_ve_lr_scalar(self):
        """MULT.RC.VE: src as LR uses the LR's value to index into R0 (combined Ra buffer)."""
        cyclic_data = [6] * 128
        r0_data = [0] * 128
        r0_data[0] = 3  # LR value 0 → Ra[0] = 3

        state = make_state(
            """\
SET lr0 cr8;;
LD.MULT r0 lr0 cr0;;
SET lr1 cr9;;
SET lr2 cr10;;
LD.MULT.CYCLIC lr1 cr0 lr2;;
MULT.RC.VE lr2 lr2 0 lr2 cr15.l cr14.l;
ACC.ADD;;
HALT;;
""",
            cr={8: 32, 9: 64, 10: 0},
        )
        write_nx8_row(state, 0x1000 // 128, r0_data)
        write_nx8_row(state, 0x2000 // 128, cyclic_data)
        run_until_complete(state)

        acc_raw = state.regfile.mutable("r_acc")
        for i in range(128):
            val = struct.unpack_from("<f", acc_raw, i * 4)[0]
            assert val == 18, f"acc word {i}: expected 18, got {val}"

    def test_mult_rc_ve_r1_scalar(self):
        """MULT.RC.VE: src LR value in [128, 255] addresses R1[value - 128] instead of R0."""
        r0_data = [0] * 128  # all zeros — must not be picked
        r1_data = [0] * 128
        r1_data[0] = 7  # LR value=128 → r1[0] = 7
        cyclic_data = [4] * 128

        state = make_state(
            """\
SET lr0 cr8;;
LD.MULT r0 lr0 cr0;;
SET lr0 cr9;;
LD.MULT r1 lr0 cr0;;
SET lr1 cr10;;
SET lr2 cr11;;
LD.MULT.CYCLIC lr1 cr0 lr2;;
SET lr3 cr12;;
MULT.RC.VE lr2 lr3 0 lr2 cr15.l cr14.l;
ACC.ADD;;
HALT;;
""",
            cr={8: 32, 9: 34, 10: 64, 11: 0, 12: 128},
        )
        write_nx8_row(state, 0x1000 // 128, r0_data)
        write_nx8_row(state, 0x1100 // 128, r1_data)
        write_nx8_row(state, 0x2000 // 128, cyclic_data)
        run_until_complete(state)

        acc_raw = state.regfile.mutable("r_acc")
        for i in range(128):
            val = struct.unpack_from("<f", acc_raw, i * 4)[0]
            assert val == 28, f"acc word {i}: expected 28 (r1[0]=7 × cyclic[i]=4), got {val}"


class TestMultRcVs:
    """MULT.RC.VS — self-multiply (square) of RC vector elements."""

    def test_mult_rc_vs_squares_rc(self):
        """Each RC element multiplied by itself (4 × 4 = 16)."""
        cyclic_data = [4] * 128

        state = make_state(
            """\
SET lr0 cr8;;
SET lr1 cr9;;
LD.MULT.CYCLIC lr0 cr0 lr1;;
SET lr5 cr9;;
MULT.RC.VS lr1 0 lr5 cr15.l cr14.l;
ACC.ADD;;
HALT;;
""",
            cr={8: 32, 9: 0},
        )
        write_nx8_row(state, 0x1000 // 128, cyclic_data)
        run_until_complete(state)

        acc_raw = state.regfile.mutable("r_acc")
        for i in range(128):
            val = struct.unpack_from("<f", acc_raw, i * 4)[0]
            assert val == 16, f"acc word {i}: expected 16 (4×4), got {val}"

    def test_mult_rc_vs_wraps_at_boundary(self):
        """rc_idx near the 512-byte boundary wraps cyclically (3 × 3 = 9)."""
        cyclic_data = [3] * 128

        state = make_state(
            """\
SET lr2 cr10;;
SET lr5 cr11;;
MULT.RC.VS lr2 0 lr5 cr15.l cr14.l;
ACC.ADD;;
HALT;;
""",
            cr={10: 450, 11: 0},
        )
        fill_nx8_cyclic(state, cyclic_data * 4)
        run_until_complete(state)

        acc_raw = state.regfile.mutable("r_acc")
        for i in range(128):
            val = struct.unpack_from("<f", acc_raw, i * 4)[0]
            assert val == 9, f"acc word {i}: expected 9 (3×3), got {val}"

    def test_mult_rc_vs_mask_zeroes_lanes(self):
        """mask_offset/mask_shift still gate elements (first 64 active, rest zeroed)."""
        cyclic_data = [3] * 128  # squared → 9
        mask_data = bytearray(128)
        for i in range(8):  # 64 bits set → first 64 lanes active
            mask_data[i] = 0xFF

        state = make_state(
            """\
SET lr0 cr8;;
SET lr1 cr9;;
LD.MULT.CYCLIC lr0 cr0 lr1;;
SET lr3 cr10;;
LD.MASK lr3 cr0;;
SET lr5 cr9;;
MULT.RC.VS lr1 0 lr5 cr15.l cr14.l;
ACC.ADD;;
HALT;;
""",
            cr={8: 32, 9: 0, 10: 64},
        )
        write_nx8_row(state, 0x1000 // 128, cyclic_data)
        state.xmem.write_address(0x2000, bytes(mask_data))
        run_until_complete(state)

        acc_raw = state.regfile.mutable("r_acc")
        for i in range(64):
            val = struct.unpack_from("<f", acc_raw, i * 4)[0]
            assert val == 9, f"word {i}: expected 9 (3×3), got {val}"
        for i in range(64, 128):
            val = struct.unpack_from("<f", acc_raw, i * 4)[0]
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

    _RC_ADDR = 0x1000  # xmem byte address for R_CYCLIC data
    _MASK_ADDR = 0x2000  # xmem byte address for mask data
    _RC_ROW = _RC_ADDR // 128  # row number for CR holding R_CYCLIC xmem base
    _MASK_ROW = _MASK_ADDR // 128  # row number for CR holding mask xmem base
    _RC_CR = 8  # CR holding R_CYCLIC xmem row
    _MASK_CR = 9  # CR holding mask xmem row
    _SHIFT_CR = 2  # CR holding mask_shift_idx (set per test)

    _ASM = """\
SET lr0 cr8;;
SET lr1 cr0;;
LD.MULT.CYCLIC lr0 cr0 lr1;;
SET lr3 cr9;;
LD.MASK lr3 cr0;;
SET lr5 cr2;;
MULT.RC.VS lr1 0 lr5 cr15.l cr14.l;
ACC.ADD;;
HALT;;
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
        state = make_state(
            self._ASM,
            cr={
                self._RC_CR: self._RC_ROW,
                self._MASK_CR: self._MASK_ROW,
                self._SHIFT_CR: shift_idx & 0xFFFFFFFF,
            },
        )
        state.set_cr_half(15, False, ds(128 // partition if partition else 128, partition or 1))
        write_nx8_row(state, self._RC_ADDR // 128, [2] * 128)
        state.xmem.write_address(self._MASK_ADDR, mask_bytes)
        run_until_complete(state)
        return state.regfile.view("r_acc")

    def _suppressed(self, acc_raw: bytes | memoryview, lane: int) -> bool:
        return struct.unpack_from("<f", acc_raw, lane * 4)[0] == 0

    # ------------------------------------------------------------------
    # shift indices with partition=0 (no partitioning — unconstrained shift)
    # ------------------------------------------------------------------

    def test_idx_0_base_mask_unchanged(self):
        """idx=0: base mask used as-is; only element 32 is active, all others suppressed."""
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
        but partition_vector[64]=0 clears it → mask = 0 → all elements deactivated.
        """
        acc = self._run_mask_shift(1 << 63, shift_idx=+1, partition=2)
        for i in range(128):
            assert self._suppressed(acc, i), f"lane {i} should be suppressed"

    def test_right_shift_stays_within_partition(self):
        """A bit inside partition 1 shifts freely within partition 1.

        With partition=2, bit 65 shifted right by 1 → bit 64 (still inside partition 1).
        inverse_partition_vector[64]=1, so the bit is NOT cleared → element 64 is active.
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
        """partition=4 (4 partitions of 32): bit at last pos of partition cannot cross → all elements deactivated."""
        acc = self._run_mask_shift(1 << 31, shift_idx=+1, partition=4)
        for i in range(128):
            assert self._suppressed(acc, i), f"lane {i} should be suppressed (partition 4)"

    def test_partition_8_boundary_constraint(self):
        """partition=8 (8 partitions of 16): bit at last pos of partition cannot cross → all elements deactivated."""
        acc = self._run_mask_shift(1 << 15, shift_idx=+1, partition=8)
        for i in range(128):
            assert self._suppressed(acc, i), f"lane {i} should be suppressed (partition 8)"

    def test_sequential_and_enforced_on_each_step(self):
        """Sequential AND is applied after EACH shift step, not just at the end.

        With partition=2, shift_idx=−2, using inverse_partition_vector (0 at end of each group):
          step 1: (1<<65 >> 1) = 1<<64; inverse_pv[64]=1 → 1<<64 (not cleared)
          step 2: (1<<64 >> 1) = 1<<63; inverse_pv[63]=0 → 0 (cleared at end of group 0)
        Result: all elements deactivated (bit cleared at second step → mask = 0).
        Without the per-step AND, a 2-bit right shift of bit 65 would land at 63
        and survive since inverse_pv[63] is never checked in a single-step AND.
        """
        acc = self._run_mask_shift(1 << 65, shift_idx=-2, partition=2)
        for i in range(128):
            assert self._suppressed(acc, i), f"lane {i} should be suppressed"


# ============================================================================
# ACTIVATE.QUANTIZE, ACTIVATE.EXACT and LD.LUT (ADR 0007)
# ============================================================================

#: The row ACTIVATE.* stores at in these tests (CR2 holds it; LR0 is 0).
_OUT_ROW = 40


def _activate(asm_fn: str, acc, *, ds_half: int | None = None, fmt=ElementFormat.NX8, **kw):
    """Run one AaQ store of R_ACC = *acc* (padded with 0); return the state."""
    acc = list(acc) + [0.0] * (128 - len(acc))
    state = make_state(f"{asm_fn};;\nHALT;;\n", cr={2: _OUT_ROW}, fmt=fmt, ds=ds_half, **kw)
    state.regfile.set_r_acc_bytes(struct.pack("<128f", *acc))
    run_until_complete(state)
    return state


def _stored_nx8(state: IpuState, fmt: Nx8Format = E4) -> list[float]:
    """The NX8 row stored at _OUT_ROW, decoded with its metadata."""
    scale, dynamic = split_metadata(state.xmem.read_metadata(_OUT_ROW * 128))
    return decode_row(bytes(state.xmem.read_address(_OUT_ROW * 128, 128)), fmt, scale, dynamic)


def _stored_fp32(state: IpuState) -> tuple[float, ...]:
    return struct.unpack("<128f", state.xmem.read_address(_OUT_ROW * 512, 512))


_QUANTIZE = "ACTIVATE.QUANTIZE {} cr15.l cr14.l cr2 lr0"
_EXACT = "ACTIVATE.EXACT {} cr15.l cr2 lr0"


class TestActivateQuantize:
    """ACTIVATE.QUANTIZE: ARCH activation, NX8 quantization, one row + metadata stored."""

    def test_identity_stores_the_quantized_row_and_its_metadata(self):
        values = [(i % 16 - 8) / 4 for i in range(128)]
        state = _activate(_QUANTIZE.format("identity"), values)
        expected = encode_row(values, E4)
        assert state.xmem.read_address(_OUT_ROW * 128, 128) == expected.elements
        assert state.xmem.read_metadata(_OUT_ROW * 128) == expected.metadata
        assert _stored_nx8(state) == values  # each value is exact in E4 at this scale

    def test_r_acc_is_not_modified(self):
        state = _activate(_QUANTIZE.format("relu"), [-42.0])
        assert struct.unpack_from("<f", state.regfile.get_r_acc_bytes())[0] == -42.0

    @pytest.mark.parametrize(
        ("fn", "x", "y"),
        [
            ("relu", -9.0, 0.0),
            ("relu", 5.0, 5.0),
            ("relu6", 7.5, 6.0),
            ("relu6", 2.0, 2.0),
            ("reciprocal", 4.0, 0.25),
            ("reciprocal", float("inf"), 0.0),
            ("rsqrt", 4.0, 0.5),
            ("rsqrt", float("inf"), 0.0),
            ("exp2", -3.0, 0.125),
            ("exp2", float("-inf"), 0.0),
            ("exp2", 0.0, 1.0),
        ],
    )
    def test_arch_functions(self, fn, x, y):
        """The seven ARCH functions and their special cases (ARCH-stage-aaq-str §4.3.1)."""
        state = _activate(_QUANTIZE.format(fn), [x], ds_half=ds(1))
        # A lone zero is "treated as normalized" (§4.3.2): it quantizes to 2^(0-127).
        assert _stored_nx8(state)[0] == pytest.approx(y, abs=1e-37)

    def test_reciprocal_of_a_subnormal_adds_epsilon(self):
        """reciprocal of 0 (a "non-negative subnormal") is 1/(0 + 2^-127) = 2^127;
        of -0, 1/(-0 - 2^-127) = -2^127. FP32 mode stores them exactly."""
        state = _activate(
            _QUANTIZE.format("reciprocal"), [0.0, -0.0], ds_half=ds(2), fmt=ElementFormat.FP32
        )
        assert _stored_fp32(state)[:2] == (2.0**127, -(2.0**127))

    def test_rsqrt_of_a_negative_input_is_nan_and_cannot_be_quantized(self):
        with pytest.raises(EmulatorError, match="NaN"):
            _activate(_QUANTIZE.format("rsqrt"), [-1.0], ds_half=ds(1))
        state = _activate(_QUANTIZE.format("rsqrt"), [-1.0], ds_half=ds(1), fmt=ElementFormat.FP32)
        assert math.isnan(_stored_fp32(state)[0])  # FP32 mode does not quantize

    def test_exp2_of_a_positive_input_is_an_emulator_error(self):
        with pytest.raises(EmulatorError, match="exp2 of a positive input"):
            _activate(_QUANTIZE.format("exp2"), [0.5], ds_half=ds(1))

    def test_generic_computes_the_declared_function(self):
        state = _activate(
            _QUANTIZE.format("generic"),
            [3.0],
            ds_half=ds(1),
            aaq=AaqOptions(generic=lambda x: x * x),
        )
        assert _stored_nx8(state)[0] == 9.0

    def test_generic_without_a_declared_function_is_an_error(self):
        with pytest.raises(EmulatorError, match="generic"):
            _activate(_QUANTIZE.format("generic"), [3.0], ds_half=ds(1))

    def test_inactive_lanes_are_zero_and_out_of_the_scale(self):
        """Only the DS's active lanes are quantized; the others are written 0 (OQ-36)."""
        values = [1.0, 2.0] + [1000.0] * 126
        state = _activate(_QUANTIZE.format("identity"), values, ds_half=ds(2))
        assert _stored_nx8(state) == [1.0, 2.0] + [0.0] * 126
        assert split_metadata(state.xmem.read_metadata(_OUT_ROW * 128))[0] == 128  # 2 = 2^1

    def test_partitions_select_the_active_lanes(self):
        """ds(3, 4 groups, 2 valid): lanes 0-2 and 32-34 are active."""
        state = _activate(_QUANTIZE.format("identity"), [1.0] * 128, ds_half=ds(3, 4, 2))
        stored = _stored_nx8(state)
        assert [i for i, v in enumerate(stored) if v] == [0, 1, 2, 32, 33, 34]

    def test_saturation_sets_quant_saturation(self):
        from ipu_isa.faults import ArithmeticEvent

        state = _activate(_QUANTIZE.format("identity"), [float("inf"), 1.0], ds_half=ds(2))
        assert ArithmeticEvent.QUANT_SATURATION in state.arith_flags
        assert _stored_nx8(state)[1] == 1.0  # +inf is left out of the scale

    def test_dynamic_format(self):
        dynamic = Nx8Format(sign=1, exp_format=0, num_bits=8, dynamic=True)
        values = [float(i % 16) / 4 for i in range(128)]
        state = _activate(_QUANTIZE.format("identity"), values, halves={"cr14.l": qf(dynamic)})
        expected = encode_row(values, dynamic)
        assert state.xmem.read_address(_OUT_ROW * 128, 128) == expected.elements
        assert state.xmem.read_metadata(_OUT_ROW * 128) == expected.metadata

    def test_fp32_mode_stores_512_bytes_and_ignores_qf(self):
        """FP32 mode: the activated row is stored as FP32, inactive lanes 0; QF is
        not read, so an invalid QF half does not matter."""
        state = _activate(
            _QUANTIZE.format("relu"),
            [-1.0, 2.5, 7.0],
            ds_half=ds(2),
            fmt=ElementFormat.FP32,
            halves={"cr14.l": 0},
        )
        assert _stored_fp32(state) == (0.0, 2.5) + (0.0,) * 126
        assert state.xmem.read_metadata(_OUT_ROW * 512) == 0

    def test_an_invalid_qf_is_refused_in_the_check_phase(self):
        """num_bits 0 is no format: the word does not retire (the LR write is undone)."""
        state = make_state(
            "SET lr5 cr3 ; " + _QUANTIZE.format("identity") + ";;\nHALT;;\n",
            cr={2: _OUT_ROW, 3: 9},
            halves={"cr14.l": 0x0140},
        )
        with pytest.raises(CrContentError, match="QF half"):
            run_until_complete(state)
        assert state.regfile.get_lr(5) == 0
        assert state.xmem.read_metadata(_OUT_ROW * 128) == 0

    def test_a_row_outside_xmem_faults_f6(self):
        state = make_state(_QUANTIZE.format("identity") + ";;\nHALT;;\n", cr={2: EIGHT_BIT_ROWS})
        with pytest.raises(EmulatorFault) as caught:
            run_until_complete(state)
        assert caught.value.fault is FaultId.F6


class TestActivateExact:
    """ACTIVATE.EXACT (emulator-only, FP32 mode only): exact activation, one FP32 row stored."""

    def _exact(self, fn: str, acc, n: int, **kw) -> tuple[float, ...]:
        return _stored_fp32(
            _activate(_EXACT.format(fn), acc, ds_half=ds(n), fmt=ElementFormat.FP32, **kw)
        )

    def test_inactive_lanes_are_written_zero(self):
        assert self._exact("relu", [-5.0, 3.0] + [99.0] * 126, 2) == (0.0, 3.0) + (0.0,) * 126

    def test_rsqrt_of_non_positive_is_zero(self):
        """ACTIVATE.EXACT keeps today's rsqrt(x <= 0) = 0 (not the ARCH NaN)."""
        assert self._exact("rsqrt", [-4.0, 0.0, 4.0], 3)[:3] == (0.0, 0.0, 0.5)

    def test_elu_respects_ipu_state_alpha(self):
        stored = self._exact("elu", [-1.0], 1, activation=ActivationParams(elu_alpha=0.5))
        assert stored[0] == pytest.approx(apply_activation(7, -1.0, elu_alpha=0.5))

    def test_window_bounds(self):
        stored = self._exact(
            "window",
            [-4.0, -3.0, 0.0, 4.0, 5.0],
            5,
            activation=ActivationParams(window_a=-3.0, window_b=5.0),
        )
        assert stored[:5] == (0.0, 1.0, 1.0, 1.0, 0.0)

    def test_sigmoid_is_exact(self):
        assert self._exact("sigmoid", [0.0], 1)[0] == 0.5

    def test_the_handler_refuses_nx8_mode(self):
        """load_program refuses the program in NX8 mode; a word written into IMEM
        directly still meets the handler's guard."""
        state = new_state()
        state.inst_mem[0] = decode_program(assemble(_EXACT.format("relu") + ";;"))[0]
        state.program_length = 1
        with pytest.raises(EmulatorError, match="FP32 mode only"):
            run_until_complete(state)


class TestLdLut:
    """LD.LUT: one row's first 128 bytes into a 1024-bit LUT segment (ADR 0007)."""

    def test_five_loads_fill_the_lut_image(self):
        from ipu_isa.lut import lut_image

        rows = lut_image(lambda x: 2.0 * x, 0, 2)
        asm = "".join(f"LD.LUT {k} cr3 lr{k};;\n" for k in range(5)) + "HALT;;\n"
        state = make_state(asm, cr={3: 10})
        for k, row in enumerate(rows):
            state.regfile.set_lr(k, k)
            state.xmem.write_address((10 + k) * 128, row)
        run_until_complete(state)
        image = int.from_bytes(state.regfile.view("aaq_lut"), "little")
        expected = 0
        for k, row in enumerate(rows[:4]):
            expected |= int.from_bytes(row, "little") << (1024 * k)
        last = int.from_bytes(rows[4], "little")
        expected |= (last & ((1 << 256) - 1)) << 4096  # table bits 4351:4096
        expected |= ((last >> 256) & ((1 << 46) - 1)) << 4352  # the LUT metadata
        assert image == expected

    def test_fp32_mode_reads_the_first_128_bytes_of_a_512_byte_row(self):
        state = make_state("LD.LUT 1 cr3 lr0;;\nHALT;;\n", cr={3: 2}, fmt=ElementFormat.FP32)
        state.xmem.write_address(2 * 512, bytes([0xAB]) * 512)
        run_until_complete(state)
        image = int.from_bytes(state.regfile.view("aaq_lut"), "little")
        assert image == int.from_bytes(bytes([0xAB]) * 128, "little") << 1024


# ============================================================================
# XMEM row addressing: .asm XMEM operands (offset + base) are row numbers,
# translated to byte addresses via the active mode's row size. XMEM is
# allocated at XMEM_SIZE_BYTES unconditionally; NX8 mode may only address
# its first EIGHT_BIT_ROWS (16384) rows -- 2 MB -- whatever the allocation size.
# ============================================================================


class TestXmemRowAddressing:
    def test_narrow_row_above_0x2000_reads_correct_bytes(self):
        """A narrow-mode row well past the old 8 KB test-suite footprint round-trips."""
        row = 100  # byte 12800 -- past 0x2000 (8192), still well inside the 2 MB narrow bound
        test_data = bytes(range(128))
        state = make_state(
            """\
SET lr13 cr8;;
LD.MULT r1 lr13 cr0;;
HALT;;
""",
            cr={8: row},
        )
        state.xmem.write_address(row * 128, test_data)
        run_until_complete(state)
        assert state.regfile.get_r(1) == bytearray(test_data)

    def test_narrow_row_near_2mb_bound_reads_correct_bytes(self):
        """The last valid narrow-mode row (16383, byte 2097024) round-trips."""
        row = EIGHT_BIT_ROWS - 1
        test_data = bytes(range(128))
        state = make_state(
            """\
SET lr13 cr8;;
LD.MULT r1 lr13 cr0;;
HALT;;
""",
            cr={8: row},
        )
        state.xmem.write_address(row * 128, test_data)
        run_until_complete(state)
        assert state.regfile.get_r(1) == bytearray(test_data)

    def test_narrow_row_at_2mb_bound_raises(self):
        """Row 16384 (the first row past the 2 MB narrow bound) is rejected in narrow mode."""
        state = make_state(
            """\
SET lr13 cr8;;
LD.MULT r1 lr13 cr0;;
HALT;;
""",
            cr={8: EIGHT_BIT_ROWS},
        )
        with pytest.raises(EmulatorFault, match="out of range in NX8 mode") as caught:
            run_until_complete(state)
        assert state.halt == HaltRecord("fault", caught.value.pc, caught.value.fault)
        assert caught.value.fault is FaultId.F6

    def test_narrow_limit_is_fixed_not_derived_from_the_allocation(self):
        """Enlarging XMEM for wide-mode kernels must not widen narrow mode's reach:
        a byte address mistaken for a row number still has to fail."""
        assert EIGHT_BIT_ROWS == 16384
        assert XMEM_SIZE_BYTES // 128 > EIGHT_BIT_ROWS

    def test_debug_mode_accepts_rows_past_the_narrow_limit(self):
        """The first row narrow mode rejects is an ordinary row in debug mode,
        which is bounded by the allocation instead."""
        state = new_state(fmt=ElementFormat.FP32)
        state.regfile.set_cr(8, EIGHT_BIT_ROWS)
        encoded = assemble("SET lr13 cr8;;\nLD.MULT r1 lr13 cr0;;\nHALT;;\n")
        load_program(state, decode_program(encoded))
        run_until_complete(state)

    def test_huge_unsigned_row_raises_narrow_range_error(self):
        """LR/CR values are unsigned 32-bit, so a 'negative' row arrives as a huge
        positive row number -- it is rejected by the narrow-mode range check, not a
        separate negative-row path (registers can never hold a negative row)."""
        state = make_state(
            """\
SET lr13 cr8;;
LD.MULT r1 lr13 cr0;;
HALT;;
""",
            cr={8: 0xFFFFFFFF},
        )
        with pytest.raises(EmulatorFault, match="out of range in NX8 mode") as caught:
            run_until_complete(state)
        assert state.halt == HaltRecord("fault", caught.value.pc, caught.value.fault)
        assert caught.value.fault is FaultId.F6

    def test_debug_mode_reaches_bytes_past_narrow_2mb_bound(self):
        """Debug mode's row size is 4x narrow's, so the same row count reaches 4x the
        bytes: row (EIGHT_BIT_ROWS - 1) narrow tops out just under 2 MB, but the same
        row number in debug mode addresses a byte well past 2 MB, deep into the
        allocation narrow mode can never reach at any row number."""
        row = EIGHT_BIT_ROWS - 2  # leaves room for a second row (cyclic data) right after
        byte_addr = row * 512
        assert byte_addr > (1 << 21), "row's debug byte address must exceed narrow's 2 MB reach"
        r0 = struct.pack("<128f", *([2.0] * 128))
        rc = struct.pack("<128f", *([3.0] * 128))
        state = new_state(fmt=ElementFormat.FP32)
        state.xmem.write_address(byte_addr, r0)
        state.xmem.write_address(byte_addr + 512, rc)
        state.regfile.set_cr(6, row)
        state.regfile.set_cr(7, row + 1)
        state.regfile.set_cr(8, 0)  # LD.MULT.CYCLIC index must be 0 in debug mode
        asm = """\
SET lr0 cr6;;
SET lr1 cr7;;
SET lr2 cr8;;
LD.MULT r0 lr0 cr0;;
LD.MULT.CYCLIC lr1 cr0 lr2;;
MULT.RC.VV lr2 r0 0 lr2 cr15.l cr14.l;;
acc.add.first;;
HALT;;
"""
        encoded = assemble(asm)
        decoded = decode_program(encoded)
        load_program(state, decoded)
        run_until_complete(state)
        acc = state.regfile.mutable("r_acc")
        for i in range(128):
            v = struct.unpack_from("<f", acc, i * 4)[0]
            assert v == pytest.approx(6.0), f"lane {i}"

    def test_debug_mode_row_past_8mb_raises(self):
        """Both modes reject rows whose byte address would exceed the XMEM allocation."""
        max_debug_row = XMEM_SIZE_BYTES // 512
        state = new_state(fmt=ElementFormat.FP32)
        state.regfile.set_cr(6, max_debug_row)
        state.regfile.set_cr(7, 0)
        asm = """\
SET lr0 cr6;;
LD.MULT r0 lr0 cr0;;
HALT;;
"""
        encoded = assemble(asm)
        decoded = decode_program(encoded)
        load_program(state, decoded)
        with pytest.raises(EmulatorFault, match="exceeds") as caught:
            run_until_complete(state)
        assert state.halt == HaltRecord("fault", caught.value.pc, caught.value.fault)
        assert caught.value.fault is FaultId.F6

    def test_ldr_mult_mask_reg_reads_row_start_128_bytes_both_modes(self):
        """LD.MASK translates the row address but always reads 128 B."""
        row = 50
        mask_data = bytes([(i + 1) & 0xFF for i in range(128)])
        state = make_state(
            """\
SET lr0 cr8;;
LD.MASK lr0 cr0;;
HALT;;
""",
            cr={8: row},
        )
        state.xmem.write_address(row * 128, mask_data)
        run_until_complete(state)
        assert state.regfile.get_r_mask() == bytearray(mask_data)


class TestNarrowStoreOverflow:
    """An NX8-mode accumulate whose float32 store overflows stores +inf and
    sets the sticky ACC_OVERFLOW flag (E04-23, Q53): an arithmetic event,
    never an exception."""

    def test_acc_add_overflow_stores_inf_and_sets_the_flag(self) -> None:
        from ipu_emu.ipu import LANES, Ipu
        from ipu_isa.faults import ArithmeticEvent

        state = make_state("ACC.ADD ;;", fmt=ElementFormat.NX8)
        k = 7
        acc = [1.0] * LANES
        mult = [2.0] * LANES
        acc[k] = mult[k] = 3e38
        state.regfile.mutable("r_acc")[:] = struct.pack(f"<{LANES}f", *acc)
        state.regfile.mutable("mult_res")[:] = struct.pack(f"<{LANES}f", *mult)

        engine = Ipu(state)
        engine.cycle.step()

        expected = [3.0] * k + [float("inf")] + [3.0] * (LANES - k - 1)
        assert state.regfile.view("r_acc")[:] == struct.pack(f"<{LANES}f", *expected)
        assert state.arith_flags == {ArithmeticEvent.ACC_OVERFLOW}


class TestFirstStoreBitExactness:
    @pytest.mark.parametrize("fmt", [ElementFormat.NX8, ElementFormat.FP32])
    @pytest.mark.parametrize("mnemonic", ["ACC.ADD.FIRST", "ACC.MAX.FIRST"])
    def test_signaling_nan_is_quietened_like_parent_lane_store(
        self,
        fmt: ElementFormat,
        mnemonic: str,
    ) -> None:
        from ipu_emu.ipu import Ipu

        state = make_state(f"{mnemonic} ;;", fmt=fmt)
        signaling_nan = struct.pack("<I", 0x7F800001)
        quiet_nan = struct.pack("<I", 0x7FC00001)
        state.regfile.mutable("mult_res")[:] = signaling_nan * 128

        engine = Ipu(state)
        engine.cycle.step()

        assert state.regfile.view("r_acc") == quiet_nan * 128


class TestInspectableSnapshot:
    def test_retained_snapshot_is_stable_and_lazily_replaced(self, monkeypatch) -> None:
        from ipu_emu.ipu import Ipu
        from ipu_emu.regfile import RegFile

        state = make_state(
            """\
SET lr0 cr8;;
SET lr0 cr9;;
HALT;;
""",
            cr={8: 1, 9: 2},
        )
        ipu = Ipu(state)

        ipu.cycle.step()
        first = ipu.snapshot
        assert first is not None
        assert ipu.snapshot is first
        assert first.get_lr(0) == 0

        copies = []
        real_snapshot = RegFile.snapshot

        def counting_snapshot(regfile):
            copies.append(regfile)
            return real_snapshot(regfile)

        monkeypatch.setattr(RegFile, "snapshot", counting_snapshot)
        ipu.cycle.step()
        # The cycle refreshes its buffer in place; no public copy until asked.
        assert copies == []
        assert first.get_lr(0) == 0
        second = ipu.snapshot
        assert len(copies) == 1
        assert second is not None
        assert second is not first
        assert second.get_lr(0) == 1


class TestDispatchPlans:
    def test_a_missing_handler_is_reported_by_instruction_name(self) -> None:
        # build_plan is the per-opcode plan the per-word schedules hold.
        import dataclasses

        from ipu_emu.ipu import build_plan
        from ipu_isa.slots import Slot
        from ipu_isa.view import isa

        fake = dataclasses.replace(
            isa().instruction(Slot.MULT, "NOP"),
            mnemonic="FAKE",
            execute_fn="execute_no_such_handler",
        )
        with pytest.raises(
            ConfigurationError,
            match=r"mult/FAKE names handler ipu_emu\.slots\.mult\.execute_no_such_handler",
        ):
            build_plan(fake, {})


# ============================================================================
# Declared slot execution order (E04-06)
# ============================================================================


class TestDeclaredSlotOrder:
    """The cycle loops follow ``ipu_isa.timing.SLOT_EXECUTION_ORDER``."""

    # BREAK stays first (it is evaluated before any write); the rest reversed.
    @staticmethod
    def _reordered():
        from ipu_isa.slots import Slot
        from ipu_isa.timing import SLOT_EXECUTION_ORDER

        rest = [slot for slot in SLOT_EXECUTION_ORDER if slot is not Slot.BREAK]
        return (Slot.BREAK, *reversed(rest))

    @pytest.mark.parametrize("skip_break", [False, True])
    def test_faithful_dispatch_follows_declared_order(self, monkeypatch, skip_break):
        import ipu_emu.cycle as cycle_mod
        import ipu_emu.ipu as ipu_mod
        from ipu_emu.issue import Issuer

        order = self._reordered()
        monkeypatch.setattr(cycle_mod, "SLOT_EXECUTION_ORDER", order)
        dispatched: list[str] = []
        real_dispatch = Issuer.dispatch
        real_lr = Issuer.dispatch_all

        def spy_dispatch(self, op, inst):
            dispatched.append(op.slot)
            return real_dispatch(self, op, inst)

        def spy_lr(self, ops, inst):
            dispatched.append("lr")
            return real_lr(self, ops, inst)

        monkeypatch.setattr(Issuer, "dispatch", spy_dispatch)
        monkeypatch.setattr(Issuer, "dispatch_all", spy_lr)

        # The BREAK.IFEQ does not fire (LR1 is 0), so the word retires.
        state = make_state("SET lr0 cr8 ; BREAK.IFEQ lr1 5;;\nHALT;;", cr={8: 7}, prototyping=False)
        engine = ipu_mod.Ipu(state)
        if skip_break:
            engine.cycle.retire()
            expected = [slot.value for slot in order[1:]]
        else:
            engine.cycle.step()
            expected = [slot.value for slot in order]
        assert dispatched == expected
        assert state.regfile.get_lr(0) == 7

    def test_prototyping_schedule_follows_declared_order(self, monkeypatch):
        import ipu_emu.cycle as cycle_mod
        import ipu_emu.ipu as ipu_mod

        order = self._reordered()
        monkeypatch.setattr(cycle_mod, "SLOT_EXECUTION_ORDER", order)
        # The NOP-skipping schedule is built and run directly, so this runs in
        # a build without --define=ipu_proto=1 as well.
        state = make_state("SET lr0 cr8;;\nHALT;;", cr={8: 7}, prototyping=False)
        schedule = cycle_mod.schedule_word(state.inst_mem[0], skip_idle_nops=True)
        seen = [op.slot for op in schedule.slot_ops]
        seen.insert(schedule.lr_position, "lr")

        expected = [slot.value for slot in order if slot.value in seen]
        assert seen == expected
        # LR and the always-run COND slot are both present, in declared order.
        assert {"lr", "cond"} <= set(seen)
        engine = ipu_mod.Ipu(state)
        cycle_mod.Cycle(state, engine, skip_idle_nops=True).step()
        assert state.regfile.get_lr(0) == 7

    def test_order_must_cover_every_planned_slot(self, monkeypatch):
        import ipu_emu.ipu as ipu_mod
        from ipu_isa.slots import Slot

        short = tuple(slot for slot in ipu_mod.SLOT_EXECUTION_ORDER if slot is not Slot.AAQ)
        monkeypatch.setattr(ipu_mod, "SLOT_EXECUTION_ORDER", short)
        with pytest.raises(ConfigurationError, match="missing \\['aaq'\\]"):
            ipu_mod._check_execution_order_covers_plans()
