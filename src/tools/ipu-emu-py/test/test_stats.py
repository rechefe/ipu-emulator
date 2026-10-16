"""Tests for RunStats counters (issue #84)."""

from __future__ import annotations

import dataclasses
from collections import Counter

import pytest
from emu_support import make_state, run

from ipu_emu.emulator import (
    DEFAULT_MAX_CYCLES,
    DebugAction,
    run_test,
    run_until_complete,
    run_with_debug,
)
from ipu_emu.errors import DebugOnlyInstructionWarning, EmulatorError, EmulatorFault
from ipu_emu.ipu_state import HaltRecord
from ipu_emu.stats import RunStats
from ipu_isa.faults import FaultId

# ---------------------------------------------------------------------------
# RunStats unit tests
# ---------------------------------------------------------------------------


class TestRunStatsProperties:
    def test_utilization_zero_cycles(self):
        s = RunStats()
        assert s.mult_utilization == 0.0
        assert s.acc_utilization == 0.0

    def test_utilization_full(self):
        s = RunStats(total_cycles=10, mult_active_cycles=10, acc_active_cycles=10)
        assert s.mult_utilization == 1.0
        assert s.acc_utilization == 1.0

    def test_utilization_partial(self):
        s = RunStats(total_cycles=4, mult_active_cycles=1, acc_active_cycles=2)
        assert s.mult_utilization == 0.25
        assert s.acc_utilization == 0.5

    def test_xmem_accesses_sum(self):
        s = RunStats(xmem_reads=3, xmem_writes=2)
        assert s.xmem_accesses == 5

    def test_format_summary_contains_key_fields(self):
        s = RunStats(
            total_cycles=100,
            mult_active_cycles=80,
            acc_active_cycles=60,
            xmem_reads=10,
            xmem_writes=5,
        )
        text = s.format_summary()
        assert "100" in text
        assert "80" in text
        assert "60" in text
        assert "10" in text
        assert "5" in text
        assert "15" in text  # xmem_accesses


# ---------------------------------------------------------------------------
# Integration: stats are populated by run_until_complete
# ---------------------------------------------------------------------------


class TestStatsCountedDuringExecution:
    def test_nop_program_has_zero_active_cycles(self):
        # All NOPs: no mult, acc, or xmem activity
        state = run("BKPT;;")
        assert state.stats.mult_active_cycles == 0
        assert state.stats.acc_active_cycles == 0
        assert state.stats.xmem_reads == 0
        assert state.stats.xmem_writes == 0

    def test_total_cycles_matches_instruction_count(self):
        # BKPT halts but counts as one cycle
        state = run("BKPT;;")
        assert state.stats.total_cycles == 1

    def test_mult_active_counted(self):
        # LDR_MULT_REG (1 read) + MULT.EE in one cycle, then BKPT
        # MULT.EE syntax: ra_idx(LR), cr_idx(CR), mask_offset(imm), mask_shift(LR), dstructure_cr_idx(CR)
        asm = """\
SET lr0 cr0;;
LDR_MULT_REG r0 lr0 cr0;MULT.EE lr0 cr0 0 lr0 cr15;;
BKPT;;
"""
        state = make_state(asm)
        state.xmem.write_address(0, bytearray(128))
        run_until_complete(state)
        assert state.stats.mult_active_cycles >= 1

    def test_acc_active_counted(self):
        asm = """\
ACC.ADD;;
BKPT;;
"""
        state = run(asm)
        assert state.stats.acc_active_cycles >= 1

    def test_xmem_read_counted(self):
        # LDR_MULT_REG reads from xmem
        asm = """\
SET lr0 cr0;;
LDR_MULT_REG r0 lr0 cr2;;
BKPT;;
"""
        state = make_state(asm)
        state.regfile.set_cr(2, 0x1000 // 128)  # row number
        state.xmem.write_address(0x1000, bytearray(128))
        run_until_complete(state)
        assert state.stats.xmem_reads == 1
        assert state.stats.xmem_writes == 0

    def test_xmem_write_counted(self):
        # STR_ACC_REG writes to xmem
        asm = """\
STR_ACC_REG lr0 cr2;;
BKPT;;
"""
        state = make_state(asm)
        state.regfile.set_lr(0, 0)
        state.regfile.set_cr(2, 0x1000 // 128)  # row number
        with pytest.warns(DebugOnlyInstructionWarning):
            run_until_complete(state)
        assert state.stats.xmem_writes == 1
        assert state.stats.xmem_reads == 0

    def test_multiple_cycles_accumulate(self):
        # 3 independent ACC cycles, then BKPT
        asm = """\
ACC.ADD;;
ACC.ADD;;
ACC.ADD;;
BKPT;;
"""
        state = run(asm)
        assert state.stats.total_cycles == 4
        assert state.stats.acc_active_cycles == 3

    def test_nop_slots_not_counted(self):
        # Three pure-LR instructions: no mult/acc/xmem
        asm = """\
SET lr0 cr0;;
ADD lr1 lr0 cr1;;
ADD lr2 lr1 cr1;;
BKPT;;
"""
        state = run(asm)
        assert state.stats.mult_active_cycles == 0
        assert state.stats.acc_active_cycles == 0
        assert state.stats.xmem_reads == 0
        assert state.stats.xmem_writes == 0


# ---------------------------------------------------------------------------
# Stats describe one run (E00-05)
# ---------------------------------------------------------------------------

_MULT_ACC_PROGRAM = """\
SET lr0 cr0;;
LDR_MULT_REG r0 lr0 cr0;MULT.EE lr0 cr0 0 lr0 cr15;;
ACC.ADD;;
LDR_MULT_REG r0 lr0 cr0;MULT.EE lr0 cr0 0 lr0 cr15;;
BKPT;;
"""

# Branches to itself forever.
_ENDLESS_PROGRAM = """\
spin:
BEQ lr0 lr0 spin;;
"""

# Two cycles complete, then the third raises an EmulatorError.
# LR0 counts up to CR8 in a two-word loop, then HALT: 2 * CR8 + 2 cycles.
_COUNTED_LOOP = """\
SET lr1 cr8;;
loop:
INC lr0 1;;
BNE lr0 lr1 loop;;
BKPT;;
"""

_FAULTING_PROGRAM = """\
SET lr0 cr8;;
SET lr1 cr9;;
LDR_CYCLIC_MULT_REG lr0 cr0 lr1;;
BKPT;;
"""


def _counters(stats: RunStats) -> dict:
    return {
        f.name: getattr(stats, f.name)
        for f in dataclasses.fields(stats)
        if f.name != "alias_profile"
    }


def _never_called(state, cycles):  # no breakpoint in these programs
    return DebugAction.CONTINUE


_RUNNERS = {
    "run_until_complete": lambda state, **kw: run_until_complete(state, **kw),
    "run_with_debug": lambda state, **kw: run_with_debug(state, _never_called, **kw),
}


class TestStatsPerRun:
    @pytest.mark.parametrize("runner", sorted(_RUNNERS))
    def test_second_run_reports_only_itself(self, runner):
        state = make_state(_MULT_ACC_PROGRAM)
        _RUNNERS[runner](state)
        first = _counters(state.stats)
        assert first["mult_active_cycles"] > 0 and first["acc_active_cycles"] > 0

        state.program_counter = 0
        _RUNNERS[runner](state)
        assert _counters(state.stats) == first
        assert state.stats.mult_utilization <= 1.0

    @pytest.mark.parametrize("runner", sorted(_RUNNERS))
    def test_cycle_limit_sets_total_cycles(self, runner):
        state = make_state(_ENDLESS_PROGRAM)
        state.stats.total_cycles = 12345  # stale value from an earlier run
        with pytest.raises(EmulatorFault, match="F-9 at PC 0: the run reached its cycle limit"):
            _RUNNERS[runner](state, max_cycles=10)
        assert state.stats.total_cycles == 10
        assert state.halt == HaltRecord("fault", 0, FaultId.F9)

    @pytest.mark.parametrize("runner", sorted(_RUNNERS))
    def test_a_limit_of_zero_never_faults(self, runner):
        # 400 cycles: past a limit of 399, and no limit at 0.
        state = make_state(_COUNTED_LOOP, cr={8: 200})
        with pytest.raises(EmulatorFault):
            _RUNNERS[runner](state, max_cycles=399)
        state = make_state(_COUNTED_LOOP, cr={8: 200})
        assert _RUNNERS[runner](state, max_cycles=0) == 402
        assert state.halt == HaltRecord("halt", 3, None) and state.cycle_limit == 0

    @pytest.mark.parametrize("runner", sorted(_RUNNERS))
    def test_reset_stats_false_accumulates(self, runner):
        state = make_state(_MULT_ACC_PROGRAM)
        cycles = _RUNNERS[runner](state)
        first = _counters(state.stats)
        state.program_counter = 0
        assert _RUNNERS[runner](state, reset_stats=False) == cycles
        numbers = {k for k, v in first.items() if isinstance(v, int | float)}
        assert {k: _counters(state.stats)[k] for k in numbers} == {k: 2 * first[k] for k in numbers}
        assert state.stats.total_cycles == 2 * cycles

    @pytest.mark.parametrize("runner", sorted(_RUNNERS))
    def test_emulator_error_sets_total_cycles(self, runner):
        state = make_state(_FAULTING_PROGRAM, cr={8: 160, 9: 64})
        state.stats.total_cycles = 12345
        with pytest.raises(EmulatorError):
            _RUNNERS[runner](state)
        assert state.stats.total_cycles == 2

    @pytest.mark.parametrize("runner", sorted(_RUNNERS))
    def test_normal_halt_sets_total_cycles(self, runner):
        state = make_state(_MULT_ACC_PROGRAM)
        cycles = _RUNNERS[runner](state)
        assert state.stats.total_cycles == cycles == 5


class TestRunStatsArithmetic:
    def _filled(self, base: int) -> RunStats:
        stats = RunStats()
        for i, f in enumerate(dataclasses.fields(stats)):
            if f.name == "alias_profile":
                continue
            if f.name == "alias_hits":
                stats.alias_hits = Counter({"A1": base, "A2": base + 1})
            else:
                setattr(stats, f.name, base + i)
        return stats

    def test_reset_zeroes_counters_and_keeps_alias_profile(self):
        profile = object()
        stats = self._filled(3)
        stats.alias_profile = profile
        hits = stats.alias_hits
        stats.reset()
        assert _counters(stats) == _counters(RunStats())
        assert stats.alias_profile is profile
        # A copy taken before the reset keeps its hits.
        assert hits == Counter({"A1": 3, "A2": 4})

    def test_reset_keeps_state_alias_profile(self):
        from ipu_emu.alias_profile import AliasProfile

        profile = AliasProfile()
        state = make_state("BKPT;;", alias_profile=profile)
        run_until_complete(state)
        assert state.stats.alias_profile is profile

    def test_add_sums_every_counter(self):
        profile = object()
        left = self._filled(1)
        left.alias_profile = profile
        right = self._filled(100)
        right.alias_profile = object()
        total = left + right
        for f in dataclasses.fields(RunStats):
            if f.name == "alias_profile":
                continue
            assert getattr(total, f.name) == getattr(left, f.name) + getattr(right, f.name), f.name
        assert total.alias_profile is profile
        # Operands are not modified.
        assert left.alias_hits == Counter({"A1": 1, "A2": 2})


def test_every_runner_defaults_to_one_cycle_limit():
    import inspect

    assert DEFAULT_MAX_CYCLES == 1_000_000
    for fn in (run_until_complete, run_with_debug, run_test):
        default = inspect.signature(fn).parameters["max_cycles"].default
        assert default == DEFAULT_MAX_CYCLES, fn.__name__
