"""Programs used to compare the Python emulator against the ETISS backend.

Each entry is a self-contained scenario: some assembly, the CR values it needs,
the data type it runs under, and optional XMEM seed data.  The parity test runs
every one of them on both backends and requires the resulting register file,
XMEM, cycle count and run statistics to be identical.

The corpus is organised by slot so that adding an instruction to
``INSTRUCTION_SPEC`` has an obvious place to gain coverage; ``test_etiss_parity``
asserts that every instruction in the spec appears in at least one program.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field


@dataclass
class Case:
    name: str
    asm: str
    cr: dict[int, int] = field(default_factory=dict)
    dtype: str = "int8"
    #: XMEM seed: {byte address: bytes}
    xmem: dict[int, bytes] = field(default_factory=dict)
    elu_alpha: float | None = None
    #: Wide-vector debug mode: 4-byte lanes, 512-byte XMEM rows.
    wide: bool = False
    #: "fp32" or "int32" -- how wide-vector lanes are interpreted.
    wide_arith: str = "fp32"
    wide_quantize: bool = False


def _ramp(n: int, start: int = 0, step: int = 1) -> bytes:
    return bytes(((start + i * step) & 0xFF) for i in range(n))


# XMEM rows are 128 bytes; row r starts at byte r * 128.
_ROW = 128

_COMMON_XMEM = {
    0 * _ROW: _ramp(128, 1, 1),        # row 0: 1,2,3,...
    1 * _ROW: _ramp(128, 255, -1),     # row 1: descending
    2 * _ROW: bytes([2]) * 128,        # row 2: constant 2
    3 * _ROW: _ramp(128, 0, 3),        # row 3
    8 * _ROW: bytes([0xFF]) * 128,     # row 8: all-ones mask source
    9 * _ROW: bytes([0x0F]) * 128,     # row 9: partial mask
    # Row 10 keeps accumulator values in [0, 5] so exp2 stays in range:
    # Python's math.exp raises OverflowError where C returns +inf, so the
    # two backends are only comparable where the reference does not raise.
    10 * _ROW: bytes((i % 6) for i in range(128)),
}


CASES: list[Case] = [
    # ---------------------------------------------------------------- LR slot
    Case(
        name="lr_set_add_sub",
        asm="""
            SET lr0 cr2 ;;
            SET lr1 cr3 ;;
            ADD lr2 lr0 lr1 ;;
            SUB lr3 lr0 lr1 ;;
            ADD lr4 lr0 cr1 ;;
        """,
        cr={2: 100, 3: 7},
    ),
    Case(
        name="lr_inc_dec",
        asm="""
            SET lr0 cr2 ;;
            INC lr0 5 ; INC lr1 1 ; DEC lr2 3 ;;
            DEC lr0 2 ;;
        """,
        cr={2: 40},
    ),
    Case(
        name="lr_three_slots_parallel",
        asm="""
            SET lr0 cr2 ; SET lr1 cr3 ; SET lr2 cr4 ;;
            ADD lr3 lr0 lr1 ; SUB lr4 lr1 lr2 ; INC lr5 9 ;;
        """,
        cr={2: 11, 3: 22, 4: 33},
    ),
    Case(
        name="lr_incr_mod_pow2",
        asm="""
            SET lr0 cr2 ;;
            INCR_MOD_POW2 lr0 cr3 4 ;;
            INCR_MOD_POW2 lr0 cr3 4 ;;
            INCR_MOD_POW2 lr0 cr3 4 ;;
            INCR_MOD_POW2 lr1 cr4 9 ;;
        """,
        cr={2: 13, 3: 5, 4: 300},
    ),
    Case(
        name="lr_addb_addbi",
        asm="""
            SET lr0 cr2 ; SET lr1 cr3 ;;
            ADDBI lrd0 5 ;;
            ADDB lrd0 cr4 ;;
            ADDBI lrd2 250 ;;
        """,
        cr={2: 0x01020304, 3: 0xF0FEFF10, 4: 0xFB},
    ),
    Case(
        name="lr_wraparound",
        asm="""
            SET lr0 cr2 ;;
            INC lr0 9 ;;
            SET lr1 cr3 ;;
            SUB lr2 lr1 lr0 ;;
        """,
        cr={2: 0xFFFFFFFA, 3: 3},
    ),

    # ------------------------------------------------------- load / store slot
    Case(
        name="load_mult_reg",
        asm="""
            SET lr0 cr0 ;;
            LDR_MULT_REG r0 lr0 cr2 ;;
            LDR_MULT_REG r1 lr0 cr3 ;;
        """,
        cr={2: 0, 3: 1},
        xmem=_COMMON_XMEM,
    ),
    Case(
        name="load_cyclic_and_mask",
        asm="""
            SET lr0 cr0 ;;
            LDR_CYCLIC_MULT_REG lr0 cr2 lr0 ;;
            SET lr1 cr4 ;;
            LDR_CYCLIC_MULT_REG lr0 cr3 lr1 ;;
            LDR_MULT_MASK_REG lr0 cr5 ;;
        """,
        cr={2: 0, 3: 1, 4: 128, 5: 9},
        xmem=_COMMON_XMEM,
    ),
    Case(
        name="store_acc_and_post_aaq",
        asm="""
            SET lr0 cr0 ;;
            LDR_MULT_REG r0 lr0 cr2 ;;
            LDR_CYCLIC_MULT_REG lr0 cr3 lr0 ;;
            MULT.RC.VV lr0 r0 0 lr0 cr15 ;;
            ACC.ADD.FIRST ;;
            STR_ACC_REG lr0 cr4 ;;
            ACTIVATE.QUANTIZE relu cr15 ;;
            STR_POST_AAQ_REG lr0 cr5 ;;
        """,
        cr={2: 0, 3: 1, 4: 32, 5: 40},
        xmem=_COMMON_XMEM,
    ),

    # -------------------------------------------------------------- mult slot
    Case(
        name="mult_rc_vv",
        asm="""
            SET lr0 cr0 ;;
            LDR_MULT_REG r0 lr0 cr2 ;;
            LDR_CYCLIC_MULT_REG lr0 cr3 lr0 ;;
            MULT.RC.VV lr0 r0 0 lr0 cr15 ;;
            ACC.ADD.FIRST ;;
            STR_ACC_REG lr0 cr4 ;;
        """,
        cr={2: 0, 3: 2, 4: 32},
        xmem=_COMMON_XMEM,
    ),
    Case(
        name="mult_rc_ve_lr_index",
        asm="""
            SET lr0 cr0 ; SET lr1 cr6 ;;
            LDR_MULT_REG r0 lr0 cr2 ;;
            LDR_CYCLIC_MULT_REG lr0 cr3 lr0 ;;
            MULT.RC.VE lr0 lr1 0 lr0 cr15 ;;
            ACC.ADD.FIRST ;;
            STR_ACC_REG lr0 cr4 ;;
        """,
        cr={2: 0, 3: 1, 4: 32, 6: 5},
        xmem=_COMMON_XMEM,
    ),
    Case(
        name="mult_rc_ve_cr_scalar",
        asm="""
            SET lr0 cr0 ;;
            LDR_CYCLIC_MULT_REG lr0 cr3 lr0 ;;
            MULT.RC.VE lr0 cr6 0 lr0 cr15 ;;
            ACC.ADD.FIRST ;;
            STR_ACC_REG lr0 cr4 ;;
        """,
        cr={3: 1, 4: 32, 6: 3},
        xmem=_COMMON_XMEM,
    ),
    Case(
        name="mult_rc_vs_square",
        asm="""
            SET lr0 cr0 ;;
            LDR_CYCLIC_MULT_REG lr0 cr3 lr0 ;;
            MULT.RC.VS lr0 0 lr0 cr15 ;;
            ACC.ADD.FIRST ;;
            STR_ACC_REG lr0 cr4 ;;
        """,
        cr={3: 0, 4: 32},
        xmem=_COMMON_XMEM,
    ),
    Case(
        name="mult_ve_rotating_window",
        asm="""
            SET lr0 cr0 ; SET lr1 cr6 ;;
            LDR_MULT_REG r0 lr0 cr2 ;;
            LDR_MULT_REG r1 lr0 cr3 ;;
            MULT.VE lr1 cr7 0 lr0 cr15 ;;
            ACC.ADD.FIRST ;;
            STR_ACC_REG lr0 cr4 ;;
        """,
        cr={2: 0, 3: 1, 4: 32, 6: 200, 7: 3},
        xmem=_COMMON_XMEM,
    ),
    Case(
        name="mult_ee_broadcast",
        asm="""
            SET lr0 cr0 ; SET lr1 cr6 ;;
            LDR_MULT_REG r0 lr0 cr2 ;;
            MULT.EE lr1 cr7 0 lr0 cr15 ;;
            ACC.ADD.FIRST ;;
            STR_ACC_REG lr0 cr4 ;;
        """,
        cr={2: 0, 4: 32, 6: 17, 7: 5},
        xmem=_COMMON_XMEM,
    ),
    Case(
        name="mult_mask_shift_partitions",
        asm="""
            SET lr0 cr0 ; SET lr1 cr6 ;;
            LDR_MULT_MASK_REG lr0 cr8 ;;
            LDR_CYCLIC_MULT_REG lr0 cr3 lr0 ;;
            MULT.RC.VS lr0 1 lr1 cr9 ;;
            ACC.ADD.FIRST ;;
            STR_ACC_REG lr0 cr4 ;;
        """,
        # CR9: valid_elements=128, partition=P4 -> (4 << 8) | 128
        cr={3: 0, 4: 32, 6: 1, 8: 9, 9: (4 << 8) | 128},
        xmem=_COMMON_XMEM,
    ),
    Case(
        name="mult_mask_shift_negative",
        asm="""
            SET lr0 cr0 ; SET lr1 cr6 ;;
            LDR_MULT_MASK_REG lr0 cr8 ;;
            LDR_CYCLIC_MULT_REG lr0 cr3 lr0 ;;
            MULT.RC.VS lr0 0 lr1 cr9 ;;
            ACC.ADD.FIRST ;;
            STR_ACC_REG lr0 cr4 ;;
        """,
        cr={3: 0, 4: 32, 6: 0xFFFFFFFE, 8: 9, 9: (8 << 8) | 128},
        xmem=_COMMON_XMEM,
    ),

    # --------------------------------------------------------------- acc slot
    Case(
        name="acc_add_max_sub_chain",
        asm="""
            SET lr0 cr0 ;;
            LDR_CYCLIC_MULT_REG lr0 cr2 lr0 ;;
            MULT.RC.VS lr0 0 lr0 cr15 ;;
            ACC.ADD.FIRST ;;
            ACC.ADD ;;
            ACC.MAX ;;
            ACC.SUB ;;
            STR_ACC_REG lr0 cr4 ;;
            ACC.MAX.FIRST ;;
            STR_ACC_REG lr0 cr5 ;;
            ACC.SUB.FIRST ;;
            STR_ACC_REG lr0 cr6 ;;
        """,
        cr={2: 0, 4: 32, 5: 36, 6: 40},
        xmem=_COMMON_XMEM,
    ),
    Case(
        name="acc_stride_variants",
        asm="""
            SET lr0 cr0 ; SET lr1 cr6 ;;
            LDR_CYCLIC_MULT_REG lr0 cr2 lr0 ;;
            MULT.RC.VS lr0 0 lr0 cr15 ;;
            ACC.STRIDE 32 on off lr0 ;;
            STR_ACC_REG lr0 cr4 ;;
            ACC.STRIDE 16 on_inv on lr1 ;;
            STR_ACC_REG lr0 cr5 ;;
            ACC.STRIDE 64 off on_inv lr0 ;;
            STR_ACC_REG lr0 cr7 ;;
        """,
        cr={2: 0, 4: 32, 5: 36, 6: 1, 7: 44},
        xmem=_COMMON_XMEM,
    ),
    Case(
        name="acc_reshape",
        asm="""
            SET lr0 cr2 ; SET lr1 cr3 ; SET lr2 cr4 ;;
            SET lr3 cr5 ;;
            LDR_CYCLIC_MULT_REG lr8 cr6 lr8 ;;
            MULT.RC.VS lr8 0 lr8 cr15 ;;
            ACC.ADD.FIRST ;;
            ACC.RESHAPE lrd0 lrd2 0 ;;
            STR_ACC_REG lr8 cr7 ;;
        """,
        cr={2: 0x03020100, 3: 0x07060504, 4: 0x0B0A0908, 5: 0x0F0E0D0C, 6: 0, 7: 32},
        xmem=_COMMON_XMEM,
    ),
    Case(
        name="agg_sum_and_max",
        asm="""
            SET lr0 cr0 ; SET lr1 cr6 ;;
            LDR_CYCLIC_MULT_REG lr0 cr2 lr0 ;;
            MULT.RC.VS lr0 0 lr0 cr15 ;;
            AGG.SUM.FIRST lr0 cr15 ;;
            AGG.SUM lr0 cr15 ;;
            AGG.MAX.FIRST lr1 cr15 ;;
            AGG.MAX lr1 cr15 ;;
            STR_ACC_REG lr0 cr4 ;;
        """,
        cr={2: 0, 4: 32, 6: 3},
        xmem=_COMMON_XMEM,
    ),
    Case(
        name="agg_partial_valid_elements",
        asm="""
            SET lr0 cr0 ;;
            LDR_CYCLIC_MULT_REG lr0 cr2 lr0 ;;
            MULT.RC.VS lr0 0 lr0 cr9 ;;
            AGG.SUM.FIRST lr0 cr9 ;;
            STR_ACC_REG lr0 cr4 ;;
        """,
        cr={2: 0, 4: 32, 9: 17},  # valid_elements = 17
        xmem=_COMMON_XMEM,
    ),

    # --------------------------------------------------------------- aaq slot
    Case(
        name="activate_all_functions",
        asm="""
            SET lr0 cr0 ;;
            LDR_CYCLIC_MULT_REG lr0 cr2 lr0 ;;
            MULT.RC.VS lr0 0 lr0 cr15 ;;
            ACC.ADD.FIRST ;;
            ACTIVATE.QUANTIZE identity cr15 ;;
            STR_POST_AAQ_REG lr0 cr4 ;;
            ACTIVATE.QUANTIZE relu cr15 ;;
            STR_POST_AAQ_REG lr0 cr5 ;;
            ACTIVATE.QUANTIZE relu6 cr15 ;;
            STR_POST_AAQ_REG lr0 cr6 ;;
            ACTIVATE.QUANTIZE sigmoid cr15 ;;
            STR_POST_AAQ_REG lr0 cr7 ;;
            ACTIVATE.QUANTIZE tanh cr15 ;;
            STR_POST_AAQ_REG lr0 cr8 ;;
            ACTIVATE.QUANTIZE gelu cr15 ;;
            STR_POST_AAQ_REG lr0 cr9 ;;
        """,
        cr={2: 0, 4: 32, 5: 36, 6: 40, 7: 44, 8: 48, 9: 52},
        xmem=_COMMON_XMEM,
    ),
    Case(
        name="activate_remaining_functions",
        asm="""
            SET lr0 cr0 ;;
            LDR_CYCLIC_MULT_REG lr0 cr2 lr0 ;;
            MULT.RC.VE lr0 cr1 0 lr0 cr15 ;;
            ACC.ADD.FIRST ;;
            ACTIVATE.QUANTIZE softplus cr15 ;;
            STR_POST_AAQ_REG lr0 cr4 ;;
            ACTIVATE.QUANTIZE elu cr15 ;;
            STR_POST_AAQ_REG lr0 cr5 ;;
            ACTIVATE.QUANTIZE exp2 cr15 ;;
            STR_POST_AAQ_REG lr0 cr6 ;;
            ACTIVATE.QUANTIZE reciprocal cr15 ;;
            STR_POST_AAQ_REG lr0 cr7 ;;
            ACTIVATE.QUANTIZE rsqrt cr15 ;;
            STR_POST_AAQ_REG lr0 cr8 ;;
            ACTIVATE.QUANTIZE silu cr15 ;;
            STR_POST_AAQ_REG lr0 cr9 ;;
        """,
        cr={2: 10, 4: 32, 5: 36, 6: 40, 7: 44, 8: 48, 9: 52},
        elu_alpha=0.5,
        xmem=_COMMON_XMEM,
    ),

    # -------------------------------------------------------------- cond slot
    Case(
        name="branch_loop_bne",
        asm="""
            SET lr0 cr0 ; SET lr1 cr2 ;;
        loop:
            INC lr0 1 ;;
            BNE lr0 lr1 loop ;;
            SET lr2 cr3 ;;
        """,
        cr={2: 12, 3: 99},
    ),
    Case(
        name="branch_beq_blt_bge",
        asm="""
            SET lr0 cr2 ; SET lr1 cr3 ;;
            BEQ lr0 lr1 equal ;;
            BLT lr0 lr1 less ;;
            SET lr2 cr4 ;;
            B done ;;
        less:
            SET lr3 cr4 ;;
            BGE lr1 lr0 ge ;;
        equal:
            SET lr4 cr4 ;;
            B done ;;
        ge:
            SET lr5 cr4 ;;
        done:
            NOP ;;
        """,
        cr={2: 3, 3: 9, 4: 77},
    ),
    Case(
        name="branch_negative_signed",
        asm="""
            SET lr0 cr2 ; SET lr1 cr3 ;;
            BLT lr0 lr1 taken ;;
            SET lr2 cr4 ;;
            B done ;;
        taken:
            SET lr3 cr4 ;;
        done:
            NOP ;;
        """,
        cr={2: 0xFFFFFFF0, 3: 1, 4: 5},  # -16 < 1 (signed)
    ),
    Case(
        name="branch_br_register",
        asm="""
            SET lr0 cr2 ;;
            BR lr0 ;;
            SET lr1 cr3 ;;
            SET lr2 cr3 ;;
            SET lr3 cr3 ;;
        """,
        cr={2: 4, 3: 42},
    ),
    Case(
        name="bkpt_halts",
        asm="""
            SET lr0 cr2 ;;
            BKPT ;;
            SET lr1 cr2 ;;
        """,
        cr={2: 8},
    ),
    Case(
        name="break_ignored_in_run_mode",
        asm="""
            SET lr0 cr2 ;;
            BREAK ;;
            INC lr0 1 ;;
            BREAK.IFEQ lr0 9 ;;
            INC lr0 1 ;;
        """,
        cr={2: 8},
    ),

    # ------------------------------------------------------------- data types
    Case(
        name="fp8_e4_multiply_accumulate",
        asm="""
            SET lr0 cr0 ;;
            LDR_MULT_REG r0 lr0 cr2 ;;
            LDR_CYCLIC_MULT_REG lr0 cr3 lr0 ;;
            MULT.RC.VV lr0 r0 0 lr0 cr15 ;;
            ACC.ADD.FIRST ;;
            MULT.RC.VS lr0 0 lr0 cr15 ;;
            ACC.ADD ;;
            ACC.MAX ;;
            STR_ACC_REG lr0 cr4 ;;
        """,
        cr={2: 0, 3: 1, 4: 32},
        dtype="fp8_e4",
        xmem=_COMMON_XMEM,
    ),
    Case(
        name="fp8_e5_agg",
        asm="""
            SET lr0 cr0 ;;
            LDR_CYCLIC_MULT_REG lr0 cr2 lr0 ;;
            MULT.RC.VS lr0 0 lr0 cr15 ;;
            AGG.SUM.FIRST lr0 cr15 ;;
            AGG.MAX lr0 cr15 ;;
            STR_ACC_REG lr0 cr4 ;;
        """,
        cr={2: 3, 4: 32},
        dtype="fp8_e5",
        xmem=_COMMON_XMEM,
    ),
    Case(
        name="fp8_e2_pad_mode_neg_inf",
        asm="""
            SET lr0 cr0 ; SET lr1 cr6 ;;
            LDR_MULT_MASK_REG lr0 cr8 ;;
            LDR_CYCLIC_MULT_REG lr0 cr2 lr0 ;;
            MULT.RC.VS lr0 0 lr1 cr9 ;;
            ACC.ADD.FIRST ;;
            STR_ACC_REG lr0 cr4 ;;
        """,
        # CR9: valid_elements=128, partition=P2, pad_mode=NEG_INF(2)
        cr={2: 0, 4: 32, 6: 1, 8: 9, 9: (2 << 13) | (2 << 8) | 128},
        dtype="fp8_e2",
        xmem=_COMMON_XMEM,
    ),

    # ------------------------------------------------- multi-slot / long runs
    Case(
        name="dense_multislot_loop",
        asm="""
            SET lr0 cr0 ; SET lr1 cr0 ; SET lr2 cr2 ;;
        loop:
            LDR_MULT_REG r0 lr1 cr3 ; MULT.RC.VV lr0 r0 0 lr0 cr15 ; ACC.ADD ;
                INC lr1 1 ; INC lr0 1 ; BNE lr1 lr2 loop ;;
            ACTIVATE.QUANTIZE relu cr15 ;;
            SET lr3 cr0 ;;
            STR_POST_AAQ_REG lr3 cr4 ;;
        """,
        cr={2: 4, 3: 0, 4: 60},
        xmem=_COMMON_XMEM,
    ),
]


# ---------------------------------------------------------------------------
# Wide-vector debug mode: 4-byte lanes, 512-byte rows, arithmetic governed by
# wide_vector_arithmetic rather than dtype.
# ---------------------------------------------------------------------------

_WIDE_ROW = 512


def _f32_row(values) -> bytes:
    return b"".join(struct.pack("<f", float(v)) for v in values)


def _i32_row(values) -> bytes:
    return b"".join(struct.pack("<i", int(v)) for v in values)


_WIDE_XMEM = {
    0 * _WIDE_ROW: _f32_row(i * 0.25 - 16.0 for i in range(128)),
    1 * _WIDE_ROW: _f32_row((i % 7) - 3 for i in range(128)),
    2 * _WIDE_ROW: _f32_row(1.0 for _ in range(128)),
    3 * _WIDE_ROW: _f32_row(0.5 * ((-1) ** i) for i in range(128)),
    8 * _WIDE_ROW: bytes([0xFF]) * 512,
}

_WIDE_INT_XMEM = {
    0 * _WIDE_ROW: _i32_row(i - 64 for i in range(128)),
    1 * _WIDE_ROW: _i32_row((i % 11) - 5 for i in range(128)),
    2 * _WIDE_ROW: _i32_row(3 for _ in range(128)),
    8 * _WIDE_ROW: bytes([0xFF]) * 512,
}

CASES += [
    Case(
        name="wide_load_and_mult_vv",
        asm="""
            SET lr0 cr0 ;;
            LDR_MULT_REG r0 lr0 cr2 ;;
            LDR_MULT_REG r1 lr0 cr3 ;;
            LDR_CYCLIC_MULT_REG lr0 cr3 lr0 ;;
            MULT.RC.VV lr0 r0 0 lr0 cr15 ;;
            ACC.ADD.FIRST ;;
            STR_ACC_REG lr0 cr4 ;;
        """,
        cr={2: 0, 3: 1, 4: 32},
        wide=True,
        xmem=_WIDE_XMEM,
    ),
    Case(
        name="wide_mult_ve_ee_vs",
        asm="""
            SET lr0 cr0 ; SET lr1 cr6 ;;
            LDR_MULT_REG r0 lr0 cr2 ;;
            LDR_CYCLIC_MULT_REG lr0 cr3 lr0 ;;
            MULT.RC.VS lr0 0 lr0 cr15 ;;
            ACC.ADD.FIRST ;;
            MULT.VE lr1 cr7 0 lr0 cr15 ;;
            ACC.ADD ;;
            MULT.EE lr1 cr7 0 lr0 cr15 ;;
            ACC.MAX ;;
            STR_ACC_REG lr0 cr4 ;;
        """,
        cr={2: 0, 3: 3, 4: 32, 6: 5, 7: 2},
        wide=True,
        xmem=_WIDE_XMEM,
    ),
    Case(
        name="wide_mult_rc_ve_scalars",
        asm="""
            SET lr0 cr0 ; SET lr1 cr6 ;;
            LDR_MULT_REG r0 lr0 cr2 ;;
            LDR_CYCLIC_MULT_REG lr0 cr3 lr0 ;;
            MULT.RC.VE lr0 lr1 0 lr0 cr15 ;;
            ACC.ADD.FIRST ;;
            MULT.RC.VE lr0 cr7 0 lr0 cr15 ;;
            ACC.SUB ;;
            STR_ACC_REG lr0 cr4 ;;
        """,
        cr={2: 0, 3: 1, 4: 32, 6: 9, 7: 3},
        wide=True,
        xmem=_WIDE_XMEM,
    ),
    Case(
        name="wide_agg_and_stride",
        asm="""
            SET lr0 cr0 ; SET lr1 cr6 ;;
            LDR_CYCLIC_MULT_REG lr0 cr2 lr0 ;;
            MULT.RC.VS lr0 0 lr0 cr15 ;;
            AGG.SUM.FIRST lr0 cr15 ;;
            AGG.SUM lr0 cr15 ;;
            AGG.MAX.FIRST lr1 cr15 ;;
            AGG.MAX lr1 cr15 ;;
            STR_ACC_REG lr0 cr4 ;;
            ACC.STRIDE 32 on off lr0 ;;
            STR_ACC_REG lr0 cr5 ;;
            ACC.RESHAPE lrd2 lrd4 0 ;;
            STR_ACC_REG lr0 cr7 ;;
        """,
        cr={2: 1, 4: 32, 5: 36, 6: 3, 7: 40},
        wide=True,
        xmem=_WIDE_XMEM,
    ),
    Case(
        name="wide_activate_no_quantize",
        asm="""
            SET lr0 cr0 ;;
            LDR_CYCLIC_MULT_REG lr0 cr2 lr0 ;;
            MULT.RC.VE lr0 cr1 0 lr0 cr15 ;;
            ACC.ADD.FIRST ;;
            ACTIVATE.QUANTIZE sigmoid cr15 ;;
            STR_POST_AAQ_REG lr0 cr4 ;;
            ACTIVATE.QUANTIZE tanh cr15 ;;
            STR_POST_AAQ_REG lr0 cr5 ;;
            ACTIVATE.QUANTIZE gelu cr15 ;;
            STR_POST_AAQ_REG lr0 cr6 ;;
        """,
        cr={2: 3, 4: 32, 5: 36, 6: 40},
        wide=True,
        xmem=_WIDE_XMEM,
    ),
    Case(
        name="wide_activate_quantized",
        asm="""
            SET lr0 cr0 ;;
            LDR_CYCLIC_MULT_REG lr0 cr2 lr0 ;;
            MULT.RC.VE lr0 cr1 0 lr0 cr15 ;;
            ACC.ADD.FIRST ;;
            ACTIVATE.QUANTIZE relu6 cr15 ;;
            STR_POST_AAQ_REG lr0 cr4 ;;
        """,
        cr={2: 1, 4: 32},
        wide=True,
        wide_quantize=True,
        xmem=_WIDE_XMEM,
    ),
    Case(
        name="wide_int32_arithmetic",
        asm="""
            SET lr0 cr0 ; SET lr1 cr6 ;;
            LDR_MULT_REG r0 lr0 cr2 ;;
            LDR_CYCLIC_MULT_REG lr0 cr3 lr0 ;;
            MULT.RC.VV lr0 r0 0 lr0 cr15 ;;
            ACC.ADD.FIRST ;;
            MULT.EE lr1 cr7 0 lr0 cr15 ;;
            ACC.ADD ;;
            AGG.SUM lr0 cr15 ;;
            STR_ACC_REG lr0 cr4 ;;
        """,
        cr={2: 0, 3: 1, 4: 32, 6: 7, 7: 5},
        wide=True,
        wide_arith="int32",
        xmem=_WIDE_INT_XMEM,
    ),
    Case(
        name="wide_mask_partitions",
        asm="""
            SET lr0 cr0 ; SET lr1 cr6 ;;
            LDR_MULT_MASK_REG lr0 cr8 ;;
            LDR_CYCLIC_MULT_REG lr0 cr2 lr0 ;;
            MULT.RC.VS lr0 0 lr1 cr9 ;;
            ACC.ADD.FIRST ;;
            STR_ACC_REG lr0 cr4 ;;
        """,
        # partition P8, pad mode +inf (representable because lanes are float)
        cr={2: 0, 4: 32, 6: 1, 8: 8, 9: (1 << 13) | (8 << 8) | 128},
        wide=True,
        xmem=_WIDE_XMEM,
    ),
]


#: Instructions the corpus intentionally does not exercise, with the reason.
UNCOVERED_OK = {
    # NOPs are exercised by every program, but never named in assembly text.
    "NOP",
}
