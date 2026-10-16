"""Which value of each datapath register a consumer reads when the same word
writes it, and the end-of-word control rules (E02-06).

"Cycle start" and "forwarded" are as in GLOSSARY.md. Each test runs one minimal
probe from ``timing_probes.py`` in faithful mode and asserts the value that
tells the versions apart. The MULT rows run in the default INT8 format and in
wide FP32 mode, whose R0/R1/R_CYCLIC reads take their own path. The two rows
decision Q32 changes are ``test_mult_reads_r_mask_forwarded`` (E04-27) and
``test_reshape_reads_previous_mult_res`` (E04-26).
"""

from __future__ import annotations

import struct

import pytest

from timing_probes import PROBES, ROW, Mode, lane, prepare, run
from ipu_emu.execute import BreakResult, Ipu
from ipu_emu.ipu_state import INST_MEM_SIZE

_MODES = pytest.mark.parametrize("mode", [Mode.NARROW, Mode.WIDE_FP32], ids=lambda m: m.value)


def _mult_res(probe: str, mode: Mode = Mode.NARROW):
    return lane(run(PROBES[probe], mode).state, "mult_res", 0)


@_MODES
def test_mult_reads_r0_r1_at_cycle_start(mode):
    """MULT reads R0/R1 at cycle start: a same-word LDR_MULT_REG (2s) is not
    seen, so R0 (1s) x CR1 = 1."""
    assert _mult_res("mult_r0_r1", mode) == 1


@_MODES
def test_mult_reads_r_cyclic_at_cycle_start(mode):
    """MULT.RC.* reads R_CYCLIC at cycle start: a same-word
    LDR_CYCLIC_MULT_REG (3s) is not seen, so 1 x 1 = 1."""
    assert _mult_res("mult_r_cyclic", mode) == 1


@_MODES
def test_mult_reads_r_mask_forwarded(mode):
    """MULT reads R_MASK forwarded: a same-word LDR_MULT_MASK_REG (all zero)
    masks lane 0, which is padded with 0 instead of holding 3 x 3. Q32 makes
    this cycle start (flipped by E04-27)."""
    assert _mult_res("mult_r_mask", mode) == 0


@_MODES
@pytest.mark.parametrize(
    ("op", "expected"),
    [
        ("add", 9),
        ("add_first", 9),
        ("sub", -9),
        ("sub_first", -9),
        ("max", 9),
        ("max_first", 9),
        ("stride", 9),
    ],
)
def test_acc_reads_this_cycle_mult_res(mode, op, expected):
    """ACC.ADD/SUB/MAX(.FIRST) and ACC.STRIDE read this cycle's MULT_RES (9s,
    not the cycle-start 1s)."""
    assert lane(run(PROBES[f"acc_mult_res_{op}"], mode).state, "r_acc", 0) == expected


@pytest.mark.parametrize(
    ("op", "expected"), [("sum", 9 * 128), ("sum_first", 9 * 128), ("max", 9), ("max_first", 9)]
)
def test_agg_reads_this_cycle_mult_res(op, expected):
    """AGG.SUM/AGG.MAX(.FIRST) read this cycle's MULT_RES (9s, not 1s)."""
    assert lane(run(PROBES[f"agg_mult_res_{op}"]).state, "r_acc", 0) == expected


def test_reshape_reads_previous_mult_res():
    """ACC.RESHAPE reads MULT_RES at cycle start: it copies 1, not this cycle's 9.
    Q32 makes this forwarded (flipped by E04-26).

    ``test_execute.py``'s ``test_reshape_mult_res_reads_from_snapshot`` pins the
    same row; E04-26 flips both or deletes the older one.
    """
    assert lane(run(PROBES["reshape_mult_res"]).state, "r_acc", 0) == 1


def test_acc_reads_old_r_acc_at_cycle_start():
    """ACC.ADD accumulates onto the cycle-start R_ACC: 5 + 9 = 14. Only the ACC
    slot writes R_ACC, so no same-word write can make the views differ today."""
    assert lane(run(PROBES["acc_old_r_acc"]).state, "r_acc", 0) == 14


@pytest.mark.parametrize(("op", "expected"), [("add_first", 7), ("add", 2 + 7)])
def test_activate_reads_this_cycle_r_acc(op, expected):
    """ACTIVATE.QUANTIZE reads this cycle's R_ACC (not the cycle-start 2).

    ``test_execute.py``'s
    ``test_agg_and_activate_quantize_same_cycle_sees_live_r_acc`` pins the AGG
    case; this one covers ACC.ADD(.FIRST).
    """
    state = run(PROBES[f"activate_r_acc_{op}"]).state
    assert state.regfile.raw("post_aaq_reg")[0] == expected


def test_store_reads_this_cycle_post_aaq():
    """STR_POST_AAQ_REG stores this cycle's POST_AAQ_REG (7, not 1)."""
    state = run(PROBES["store_post_aaq"]).state
    assert state.xmem.read_address(ROW * 128, 1) == b"\x07"


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_acc_store_reads_this_cycle_r_acc():
    """STR_ACC_REG stores this cycle's R_ACC (7, not 1)."""
    state = run(PROBES["acc_store_r_acc"]).state
    assert struct.unpack("<i", state.xmem.read_address(ROW * 128, 4))[0] == 7


def test_bkpt_halts_after_whole_word_retires():
    """BKPT halts after every other slot of its word has run: the LR write and
    the store land, and the next word does not run. (BKPT becomes HALT in E03.)"""
    result = run(PROBES["bkpt"])
    state = result.state
    assert (
        result.cycles,
        state.program_counter,
        state.regfile.get_lr(0),
        state.xmem.read_address(ROW * 128, 1),
    ) == (1, INST_MEM_SIZE, 5, b"\x07")


@pytest.mark.parametrize(("probe", "marker"), [("signed_blt", 2), ("signed_bge", 1)])
def test_branches_compare_signed(probe, marker):
    """BLT and BGE compare signed 32-bit values: 0xFFFFFFFF is -1 < 0. (They
    become BLT.S/BGE.S in E03.)"""
    assert run(PROBES[probe]).state.regfile.get_lr(7) == marker


def test_break_has_no_side_effects_and_resume_rereads():
    """A BREAK word stops before any slot runs. Resuming re-reads the registers,
    so an edit made while stopped is what the word sees."""
    state = prepare(PROBES["break_resume"])
    ipu = Ipu(state)
    assert ipu.execute_vliw_cycle() == BreakResult.BREAK
    assert (state.program_counter, state.regfile.get_lr(0)) == (0, 0)
    state.regfile.set_cr(2, 9)
    ipu.execute_vliw_cycle_skip_break()
    assert (state.program_counter, state.regfile.get_lr(0)) == (1, 9)


def test_idle_word_advances_pc_once():
    """An empty instruction-memory entry advances the PC by one and counts one
    idle MULT cycle, and runs nothing."""
    state = prepare(PROBES["idle_word"])
    ipu = Ipu(state)
    ipu.execute_vliw_cycle()
    idle_before = state.stats.mult_idle_cycles
    ipu.execute_vliw_cycle()
    assert (
        state.program_counter,
        state.stats.mult_idle_cycles - idle_before,
        state.regfile.get_lr(1),
    ) == (2, 1, 0)
