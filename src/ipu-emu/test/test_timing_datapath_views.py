"""Which value of each datapath register a consumer reads when the same word
writes it, and the end-of-word control rules (E02-06).

"Cycle start" and "forwarded" are as in GLOSSARY.md. Each test runs one minimal
probe from ``timing_probes.py`` in faithful mode and asserts the value that
tells the versions apart. The MULT rows run in NX8 mode and in FP32 mode,
whose R0/R1/R_CYCLIC reads take their own path. The two rows
decision Q32 changed are ``test_mult_reads_r_mask_at_cycle_start`` (E04-27) and
``test_reshape_reads_this_cycle_mult_res`` (E04-26).
"""

from __future__ import annotations

import pytest
from timing_probes import PROBES, ROW, Mode, lane, prepare, run, stored_element

from ipu_emu import prototyping as prototyping_mode
from ipu_emu.ipu import Ipu
from ipu_emu.ipu_state import INST_MEM_SIZE
from ipu_emu.outcome import CycleOutcome

_MODES = pytest.mark.parametrize("mode", [Mode.NX8, Mode.WIDE_FP32], ids=lambda m: m.value)


def _mult_res(probe: str, mode: Mode = Mode.NX8):
    return lane(run(PROBES[probe], mode).state, "mult_res", 0)


@_MODES
@pytest.mark.parametrize("probe", ["mult_ra_operand", "mult_r0_r1"])
def test_mult_reads_r0_r1_at_cycle_start(mode, probe):
    """MULT reads R0/R1 at cycle start, through the ``ra`` operand (MULT.RC.VV)
    and inside MULT.VE alike: a same-word LD.MULT (2s) is not seen, so
    R0 (1s) x 1 = 1."""
    assert _mult_res(probe, mode) == 1


@_MODES
def test_mult_reads_r_cyclic_at_cycle_start(mode):
    """MULT.RC.* reads R_CYCLIC at cycle start: a same-word
    LD.MULT.CYCLIC (3s) is not seen, so 1 x 1 = 1."""
    assert _mult_res("mult_r_cyclic", mode) == 1


@_MODES
@pytest.mark.parametrize("prototyping", [False, True], ids=["faithful", "prototyping"])
def test_mult_reads_r_mask_at_cycle_start(mode, prototyping):
    """MULT reads R_MASK at cycle start (Q20, flipped by E04-27): a same-word
    LD.MASK (all zero) is not seen, so lane 0 holds 3 x 3 under the reset
    all-ones mask. One word later the multiply sees the new mask and pads
    lane 0 with 0. The prototyping all-ones shortcut reads the same view."""
    if prototyping and not prototyping_mode.build_enabled():
        pytest.skip("prototyping needs --config=proto")
    same_word = run(PROBES["mult_r_mask"], mode, prototyping=prototyping).state
    next_word = run(PROBES["mult_r_mask_next_word"], mode, prototyping=prototyping).state
    assert lane(same_word, "mult_res", 0) == 9
    assert lane(next_word, "mult_res", 0) == 0


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


def test_reshape_reads_this_cycle_mult_res():
    """ACC.RESHAPE reads this cycle's MULT_RES (9s, not 1s), like every other
    ACC-stage op (Q32 rule 3; E04-26 flipped it from cycle start).

    ``test_execute.py``'s ``test_reshape_reads_same_cycle_mult_res`` pins the
    same row with a scatter of distinct products.
    """
    assert lane(run(PROBES["reshape_mult_res"]).state, "r_acc", 0) == 9


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
    assert stored_element(state, ROW, 0) == expected


def test_activate_exact_reads_this_cycle_r_acc():
    """ACTIVATE.EXACT (FP32 mode) stores this cycle's R_ACC (7, not 1)."""
    state = run(PROBES["exact_r_acc"], Mode.WIDE_FP32).state
    assert stored_element(state, ROW, 0) == 7


def test_bkpt_halts_after_whole_word_retires():
    """HALT halts after every other slot of its word has run: the LR write and
    the AaQ store land, and the next word does not run."""
    result = run(PROBES["halt"])
    state = result.state
    assert (
        result.cycles,
        state.program_counter,
        state.regfile.get_lr(0),
        stored_element(state, ROW, 0),
    ) == (1, INST_MEM_SIZE, 5, 7)


@pytest.mark.parametrize(("probe", "marker"), [("signed_blt", 2), ("signed_bge", 1)])
def test_branches_compare_signed(probe, marker):
    """BLT.S and BGE.S compare signed 32-bit values: 0xFFFFFFFF is -1 < 0. (They
    become BLT.S/BGE.S in E03.)"""
    assert run(PROBES[probe]).state.regfile.get_lr(7) == marker


def test_break_has_no_side_effects_and_resume_rereads():
    """A BREAK word stops before any slot runs. Resuming re-reads the registers,
    so an edit made while stopped is what the word sees."""
    state = prepare(PROBES["break_resume"])
    ipu = Ipu(state)
    assert ipu.cycle.step() == CycleOutcome.BREAK
    assert (state.program_counter, state.regfile.get_lr(0)) == (0, 0)
    state.regfile.set_cr(2, 9)
    ipu.cycle.retire()
    assert (state.program_counter, state.regfile.get_lr(0)) == (1, 9)


def test_idle_word_advances_pc_once():
    """An empty instruction-memory entry advances the PC by one and counts one
    idle MULT cycle, and runs nothing."""
    state = prepare(PROBES["idle_word"])
    ipu = Ipu(state)
    ipu.cycle.step()
    idle_before = state.stats.mult_idle_cycles
    ipu.cycle.step()
    assert (
        state.program_counter,
        state.stats.mult_idle_cycles - idle_before,
        state.regfile.get_lr(1),
    ) == (2, 1, 0)
