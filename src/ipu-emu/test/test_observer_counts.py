"""A cycle observer counts the same ops in faithful and prototyping mode (E04-11).

An observer attached with ``IpuState.add_cycle_observer`` sees each op the one
cycle module issues, whichever schedule drives it. Prototyping's schedule
skips idle NOP slots (``skip_idle_nops``), which a count of non-NOP ops
ignores; everything else is the same cycle. A default build drives the
prototyping schedule through ``Cycle(..., skip_idle_nops=True)``; a
``--config=proto`` build also runs the real prototyping mode end to end.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
from emu_support import assemble_program, new_state

from ipu_emu.cycle import Cycle, CycleOutcome
from ipu_emu.emulator import load_program, run_until_complete
from ipu_emu.ipu import Ipu
from ipu_emu.ipu_state import IpuState
from ipu_emu.issue import IssueFacts
from ipu_emu.prototyping import build_enabled
from ipu_isa.slots import Slot

#: The slots counted: every slot but LR.
_COUNTED = frozenset(slot.value for slot in Slot if slot is not Slot.LR)

#: Every slot kind and a BREAK in a loop: loads, multiplies, accumulates,
#: aggregation, the activation, a store, LR arithmetic and the cond slot.
_ALL_SLOTS_ASM = """
    SET LR0 CR0 ; SET LR1 CR1 ;;
    LD.MULT R0 LR0 CR2 ;;
    LD.MULT.CYCLIC LR0 CR3 LR0 ;;
loop:
    MULT.RC.VV LR0 R0 0 LR0 CR15 ; ACC.ADD.FIRST ;;
    MULT.RC.VE LR0 CR1 0 LR0 CR15 ; ACC.ADD ; BREAK ;;
    MULT.EE LR0 CR1 0 LR0 CR15 ; AGG.SUM.FIRST LR1 CR15 ;;
    ACTIVATE.QUANTIZE relu CR15 ; ST.AAQ LR1 CR4 ; INC LR2 1 ;;
    BLT.S LR2 CR7 loop ;;
    HALT ;;
"""

#: A word with a BREAK, so a cycle stops and resumes; mostly idle slots.
_BREAKING_ASM = """
loop:
    INC LR1 1 ; BREAK ;;
    BLT.S LR1 CR8 loop ;;
    HALT ;;
"""


def _all_slots_setup(state: IpuState) -> None:
    rng = np.random.RandomState(7)
    state.xmem.write_address(0x10000, bytes(rng.randint(0, 256, 128 * 4, dtype=np.uint8)))
    state.regfile.set_cr(2, 0x10000 // 128)  # R0's source row
    state.regfile.set_cr(3, 0x10000 // 128 + 1)  # the cyclic source row
    state.regfile.set_cr(4, 0x30000 // 128)  # the store row
    state.regfile.set_cr(7, 3)  # the loop bound
    state.set_cr_dstructure(valid_elements=128, partition=4)


def _breaking_setup(state: IpuState) -> None:
    state.regfile.set_cr(8, 4)


_PROGRAMS = {
    "all_slots": (_ALL_SLOTS_ASM, _all_slots_setup),
    "break_and_resume": (_BREAKING_ASM, _breaking_setup),
}


@dataclass
class _Count:
    """Issued non-NOP ops by slot: a :class:`~ipu_emu.cycle.CycleObserver`."""

    by_slot: dict[str, int] = field(default_factory=dict)

    def begin(self, pc: int) -> None:
        pass

    def issued(self, facts: IssueFacts) -> None:
        if facts.slot in _COUNTED and facts.plan.inst_name != "NOP":
            self.by_slot[facts.slot] = self.by_slot.get(facts.slot, 0) + 1

    def retired(self, facts: IssueFacts, result: Any) -> None:
        pass

    def end(self) -> None:
        pass


@contextmanager
def _counting(state: IpuState) -> Iterator[_Count]:
    count = _Count()
    remove = state.add_cycle_observer(count)
    try:
        yield count
    finally:
        remove()


def _state(program: str, *, prototyping: bool = False) -> IpuState:
    asm, setup = _PROGRAMS[program]
    state = new_state(setup=setup, prototyping=prototyping)
    load_program(state, assemble_program(asm))
    return state


def _run_schedule(state: IpuState, *, skip_idle_nops: bool) -> None:
    """Run *state* to HALT on a cycle with the given schedule, resuming at each BREAK."""
    cycle = Cycle(state, Ipu(state), skip_idle_nops=skip_idle_nops)
    for _ in range(100_000):
        if state.is_halted:
            break
        if cycle.step() is CycleOutcome.BREAK:
            cycle.retire()
    assert state.is_halted


@pytest.mark.parametrize("program", sorted(_PROGRAMS))
def test_counts_are_identical_in_faithful_and_prototyping_mode(program: str):
    counts = {}
    for name, skip in (("faithful", False), ("prototyping schedule", True)):
        state = _state(program)
        with _counting(state) as counts[name]:
            _run_schedule(state, skip_idle_nops=skip)
        assert state.cycle_observers == ()  # detached on exit
    state = _state(program)
    with _counting(state) as counts["run_until_complete"]:
        run_until_complete(state, max_cycles=100_000)
    if build_enabled():
        state = _state(program, prototyping=True)
        with _counting(state) as counts["prototyping"]:
            run_until_complete(state, max_cycles=100_000)

    reference = counts["faithful"].by_slot
    assert reference and sum(reference.values()) > 0
    assert {name: c.by_slot for name, c in counts.items()} == dict.fromkeys(counts, reference)
    if program == "break_and_resume":
        assert reference["break"] == 4
