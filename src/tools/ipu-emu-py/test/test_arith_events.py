"""Arithmetic events (E04-23, Q53): an accumulator overflow or NaN result
stores the IEEE result and sets a sticky flag; it never raises or halts."""

from __future__ import annotations

import math
import struct

import numpy as np
import pytest
from emu_support import ElementFormat, make_state

from ipu_emu.commit import CommitCollector
from ipu_emu.emulator import run_until_complete
from ipu_emu.ipu_state import HaltRecord
from ipu_isa.faults import ArithmeticEvent
from ipu_isa.geometry import LANES

OVERFLOW, NAN = ArithmeticEvent.ACC_OVERFLOW, ArithmeticEvent.NAN_RESULT


def _floats(state, name, values):
    state.regfile.mutable(name)[:] = struct.pack(f"<{LANES}f", *values)


def _acc_floats(state):
    return list(struct.unpack(f"<{LANES}f", state.regfile.view("r_acc")))


def test_fp32_overflow_is_inf_in_its_lane_only():
    state = make_state("ACC.ADD ;;\nHALT ;;", fmt=ElementFormat.FP32)
    acc, res = [1.0] * LANES, [2.0] * LANES
    acc[9] = res[9] = 3e38
    _floats(state, "r_acc", acc)
    _floats(state, "mult_res", res)
    run_until_complete(state)
    got = _acc_floats(state)
    assert got[9] == math.inf and got[:9] == [3.0] * 9 and got[10:] == [3.0] * (LANES - 10)
    assert state.arith_flags == {OVERFLOW} and state.halt == HaltRecord("halt", 1, None)


def test_int8_accumulate_wraps_at_two_to_the_31_and_sets_the_flag():
    state = make_state("ACC.ADD ;;\nHALT ;;", fmt=ElementFormat.INT8)
    for lane in range(LANES):
        state.regfile.set_r_acc_word(lane, 2**31 - 1 if lane == 3 else 5)
    state.regfile.mutable("mult_res")[:] = struct.pack(f"<{LANES}i", *([1] * LANES))
    run_until_complete(state)
    lanes = np.frombuffer(state.regfile.view("r_acc"), dtype="<i4")
    assert lanes[3] == -(2**31) and (np.delete(lanes, 3) == 6).all()
    assert state.arith_flags == {OVERFLOW}


def test_a_nan_input_propagates_and_sets_the_flag():
    state = make_state("ACC.ADD ;;\nHALT ;;", fmt=ElementFormat.FP32)
    res = [1.0] * LANES
    res[0] = math.nan
    _floats(state, "r_acc", [0.0] * LANES)
    _floats(state, "mult_res", res)
    run_until_complete(state)
    got = _acc_floats(state)
    assert math.isnan(got[0]) and got[1:] == [1.0] * (LANES - 1)
    assert state.arith_flags == {NAN}


def test_an_infinite_input_is_no_overflow():
    # inf + 1 is inf: an IEEE result from an infinite input, not an overflow.
    state = make_state("ACC.ADD ;;\nHALT ;;", fmt=ElementFormat.FP32)
    res = [1.0] * LANES
    res[0] = math.inf
    _floats(state, "r_acc", [0.0] * LANES)
    _floats(state, "mult_res", res)
    run_until_complete(state)
    assert state.arith_flags == set()


def test_flags_are_sticky_across_cycles_and_runs():
    state = make_state("ACC.ADD ;;\nACC.ADD.FIRST ;;\nHALT ;;", fmt=ElementFormat.FP32)
    acc, res = [1.0] * LANES, [1.0] * LANES
    acc[0] = res[0] = 3e38
    _floats(state, "r_acc", acc)
    _floats(state, "mult_res", res)
    run_until_complete(state)
    assert state.arith_flags == {OVERFLOW}  # the later clean word does not clear it
    state.program_counter = 1
    run_until_complete(state)
    assert state.arith_flags == {OVERFLOW}  # nor does a new run


def test_agg_sum_overflow_sets_the_flag():
    state = make_state("AGG.SUM.FIRST LR0 CR15 ;;\nHALT ;;", fmt=ElementFormat.FP32)
    _floats(state, "mult_res", [3e38] * LANES)
    run_until_complete(state)
    assert _acc_floats(state)[0] == math.inf and state.arith_flags == {OVERFLOW}


def test_the_commit_record_lists_the_word_s_events():
    state = make_state("ACC.ADD ;;\nACC.ADD ;;\nHALT ;;", fmt=ElementFormat.FP32)
    acc, res = [1.0] * LANES, [0.0] * LANES
    res[0] = 3e38
    acc[0] = 3e38
    _floats(state, "r_acc", acc)
    _floats(state, "mult_res", res)
    with CommitCollector(state) as collector:
        run_until_complete(state)
    # Word 0: 3e38 + 3e38 overflows. Word 1: inf + 3e38 is inf, no new event.
    assert [r.arith_events for r in collector.records] == [("acc_overflow",), (), ()]


@pytest.mark.parametrize("fmt", [ElementFormat.FP32, ElementFormat.E4, ElementFormat.INT8])
def test_a_clean_accumulate_reports_nothing(fmt):
    state = make_state("ACC.ADD ;;\nHALT ;;", fmt=fmt)
    run_until_complete(state)
    assert state.arith_flags == set()
