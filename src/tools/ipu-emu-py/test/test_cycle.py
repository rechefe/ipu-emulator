"""The Cycle module (E04-07): one implementation of one VLIW word.

``step()`` is ``evaluate_break()`` then ``retire()``; ``evaluate_break()`` has no
side effects; an idle word counts one idle multiplier cycle; ``retire()`` takes
the snapshot again, so a debugger's edit made at a break is visible.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
from emu_support import assemble_program, make_state, new_state

from ipu_emu.cycle import Cycle, CycleOutcome, decode_word, schedule_word
from ipu_emu.emulator import load_program
from ipu_emu.execute import BreakResult, Ipu
from ipu_emu.ipu_state import IpuState
from ipu_emu.regfile import RegFile

# Every slot kind, a BREAK in the middle and a branch back: loads, multiplies,
# accumulates, aggregation, a store, LR arithmetic and the cond slot.
PROGRAM = """
    SET LR0 CR0 ; SET LR1 CR1 ;;
    LDR_MULT_REG R0 LR0 CR2 ;;
    LDR_CYCLIC_MULT_REG LR0 CR3 LR0 ;;
loop:
    MULT.RC.VV LR0 R0 0 LR0 CR15 ; ACC.ADD.FIRST ;;
    MULT.RC.VE LR0 CR1 0 LR0 CR15 ; ACC.ADD ; BREAK ;;
    MULT.EE LR0 CR1 0 LR0 CR15 ; AGG.SUM.FIRST LR1 CR15 ;;
    ACTIVATE.QUANTIZE relu CR15 ; STR_POST_AAQ_REG LR1 CR4 ; INC LR2 1 ;;
    BLT LR2 CR7 loop ;;
    BKPT ;;
"""

STORE_ROW_ADDRESS = 0x30000


def _seed(state: IpuState) -> None:
    rng = np.random.RandomState(7)
    state.xmem.write_address(0x10000, bytes(rng.randint(0, 256, 128 * 4, dtype=np.uint8)))
    state.regfile.set_cr(2, 0x10000 // 128)  # R0 source row
    state.regfile.set_cr(3, 0x10000 // 128 + 1)  # cyclic source row
    state.regfile.set_cr(4, STORE_ROW_ADDRESS // 128)  # post-AAQ store row
    state.regfile.set_cr(7, 3)  # loop bound
    state.set_cr_dstructure(valid_elements=128, partition=4)


def _program_state(asm: str = PROGRAM, **kw) -> IpuState:
    state = new_state(setup=_seed, **kw)
    load_program(state, assemble_program(asm))
    return state


def _observable(state: IpuState) -> tuple:
    """Everything a word can change: PC, every register, the store row, stats."""
    return (
        state.program_counter,
        {name: bytes(state.regfile.raw_readonly(name)) for name in state.regfile.register_names},
        state.xmem.snapshot_bytes(STORE_ROW_ADDRESS, 128),
        copy.deepcopy(state.stats),
    )


def _cycle(state: IpuState, observer=None) -> Cycle:
    return Cycle(state, Ipu(state), observer)


class _Recorder:
    """A CycleObserver that records its calls."""

    def __init__(self, state: IpuState) -> None:
        self.state = state
        self.calls: list[tuple] = []

    def begin(self, pc: int) -> None:
        self.calls.append(("begin", pc))

    def end(self) -> None:
        self.calls.append(("end", self.state.program_counter))


def test_step_equals_evaluate_break_then_retire():
    stepped, split = _program_state(), _program_state()
    by_step, by_parts = _cycle(stepped), _cycle(split)
    outcomes = []
    while not stepped.is_halted:
        outcome = by_step.step()
        if outcome is CycleOutcome.BREAK:
            by_step.retire()
        assert by_parts.evaluate_break() is outcome
        by_parts.retire()
        outcomes.append(outcome)
        assert _observable(stepped) == _observable(split)
    assert split.is_halted
    assert outcomes.count(CycleOutcome.BREAK) == 3  # once per loop iteration
    assert len(outcomes) > 8


@pytest.mark.parametrize("fires", [True, False])
def test_evaluate_break_has_no_side_effects(fires):
    # The word loads, multiplies, accumulates, stores and writes LRs.
    breaking = "BREAK" if fires else "BREAK.IFEQ LR1 5"
    asm = (
        "LDR_MULT_REG R0 LR0 CR2 ; MULT.RC.VV LR0 R0 0 LR0 CR15 ; ACC.ADD.FIRST ; "
        f"STR_POST_AAQ_REG LR0 CR4 ; SET LR0 CR3 ; INC LR3 2 ; {breaking} ;;\nBKPT ;;"
    )
    state, twin = _program_state(asm), _program_state(asm)
    before = _observable(twin)

    outcome = _cycle(state).evaluate_break()

    assert outcome is (CycleOutcome.BREAK if fires else CycleOutcome.RETIRED)
    assert _observable(state) == before
    assert state.xmem == twin.xmem


def test_an_idle_word_counts_one_idle_cycle_and_advances_the_pc():
    state = new_state()
    load_program(state, [None, *assemble_program("BKPT ;;")])
    cycle = _cycle(state)

    assert cycle.step() is CycleOutcome.RETIRED

    assert state.program_counter == 1
    assert state.stats.mult_idle_cycles == 1
    assert state.stats.mult_active_cycles == 0


def test_an_idle_word_split_at_its_break_still_counts_once():
    state = new_state()
    load_program(state, [None, *assemble_program("BKPT ;;")])
    cycle = _cycle(state)

    assert cycle.evaluate_break() is CycleOutcome.RETIRED
    assert (state.program_counter, state.stats.mult_idle_cycles) == (0, 0)
    cycle.retire()
    assert (state.program_counter, state.stats.mult_idle_cycles) == (1, 1)


def test_without_honour_break_the_word_retires():
    state = make_state("SET LR0 CR8 ; BREAK ;;\nBKPT ;;", cr={8: 42})
    cycle = _cycle(state)

    assert cycle.step(honour_break=False) is CycleOutcome.RETIRED

    assert state.regfile.get_lr(0) == 42
    assert state.program_counter == 1


def test_a_break_stops_before_any_side_effect():
    state = make_state("SET LR0 CR8 ; BREAK ;;\nBKPT ;;", cr={8: 42})
    cycle = _cycle(state)

    assert cycle.step() is CycleOutcome.BREAK

    assert state.regfile.get_lr(0) == 0
    assert state.program_counter == 0


def test_an_edit_made_at_a_break_is_visible_to_the_resumed_word():
    state = make_state("SET LR0 CR8 ; ADD LR1 LR2 CR8 ; BREAK ;;\nBKPT ;;", cr={8: 42})
    cycle = _cycle(state)
    assert cycle.evaluate_break() is CycleOutcome.BREAK

    # What a debugger does while stopped at the break.
    state.regfile.set_cr(8, 7)
    state.regfile.set_lr(2, 100)
    cycle.retire()

    assert state.regfile.get_lr(0) == 7
    assert state.regfile.get_lr(1) == 107


def test_reset_empties_the_schedule_cache_and_recopies_the_register_file(monkeypatch):
    state = make_state("SET LR0 CR8 ;;\nBKPT ;;", cr={8: 1})
    ipu = Ipu(state)
    ipu.cycle.step()
    # TODO(E04-08): the per-word plan cache moves into Cycle; assert through it.
    ipu._cycle_plan(0, state.inst_mem[0])
    assert ipu._cycle_cache

    copies = []
    real_snapshot = RegFile.snapshot

    def counting_snapshot(regfile):
        copies.append(regfile)
        return real_snapshot(regfile)

    monkeypatch.setattr(RegFile, "snapshot", counting_snapshot)
    ipu.cycle.reset()

    assert not ipu._cycle_cache
    ipu.cycle.step()
    assert copies == [state.regfile]


def test_the_observer_brackets_each_retired_word_and_not_a_break():
    state = make_state("SET LR0 CR8 ; BREAK ;;\nSET LR1 CR8 ;;\nBKPT ;;", cr={8: 3})
    recorder = _Recorder(state)
    cycle = _cycle(state, recorder)

    assert cycle.step() is CycleOutcome.BREAK
    assert recorder.calls == []
    cycle.retire()
    cycle.step()
    assert recorder.calls == [("begin", 0), ("end", 1), ("begin", 1), ("end", 2)]


def test_an_lr_conflict_raises_before_any_lr_write():
    state = make_state("SET LR0 CR8 ; SET LR0 CR9 ; INC LR3 1 ;;\nBKPT ;;", cr={8: 5, 9: 6})
    with pytest.raises(RuntimeError, match=r"LR conflict.*targets: \[0, 0, 3\]"):
        _cycle(state).step()
    assert state.regfile.get_lr(0) == 0
    assert state.regfile.get_lr(3) == 0


def test_schedule_of_a_word():
    word = assemble_program("SET LR0 CR1 ; INC LR2 1 ; ACC.ADD ; BREAK ;;")[0]

    schedule = schedule_word(word)

    assert schedule.break_op is not None
    assert schedule.break_op.plan.inst_name == "BREAK"
    assert schedule.lr_conflict is None
    assert [op.plan.inst_name for op in schedule.lr_ops if op.plan.inst_name != "NOP"] == [
        "SET",
        "INC",
    ]
    assert [op.slot for op in schedule.slot_ops] == [
        op.slot for op in decode_word(word) if op.slot not in ("break", "lr")
    ]
    assert "acc" in {op.slot for op in schedule.slot_ops if op.plan.inst_name != "NOP"}


def test_a_nop_break_slot_is_not_scheduled_and_conflicts_are_found_once():
    word = assemble_program("SET LR0 CR1 ; SET LR0 CR2 ;;")[0]

    schedule = schedule_word(word)

    assert schedule.break_op is None
    assert schedule.lr_conflict == (0, 0)


def test_break_result_is_an_alias_of_cycle_outcome():
    assert BreakResult is CycleOutcome
    assert BreakResult.CONTINUE is CycleOutcome.RETIRED
