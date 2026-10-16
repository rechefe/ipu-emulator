"""One run's cycle count, its limit (fault F-9) and its stats totals (E04-21).

Both runners in :mod:`ipu_emu.emulator` drive a :class:`RunClock`: it starts
the run, admits each cycle against ``IpuState.cycle_limit`` and writes
``RunStats.total_cycles`` when the run ends, however it ends. The steppable
engine owns one once it lands. (Not ``RunControl``: that name is the
debugger's seam in :mod:`ipu_emu.run_control`.)
"""

from __future__ import annotations

from ipu_emu.errors import EmulatorFault
from ipu_emu.ipu_state import HaltRecord, IpuState
from ipu_isa.faults import FaultId

__all__ = ["RunClock"]


class RunClock:
    """Counts the cycles of one run on ``state``.

    Args:
        state: The machine state the run executes on.
        limit: ``state.cycle_limit`` for this run only (restored by
            :meth:`end_run`); ``None`` keeps the state's own.

    Attributes:
        cycles: Cycles this run has completed.
    """

    def __init__(self, state: IpuState, limit: int | None = None) -> None:
        self.state = state
        self.cycles = 0
        self._earlier = 0  # total_cycles carried in when stats accumulate
        self._limit = limit
        self._own_limit = state.cycle_limit

    def begin_run(self, *, reset_stats: bool = True) -> None:
        """Start a run: no halt or quit pending, no cycles counted.

        Args:
            reset_stats: Start ``state.stats`` from zero, so it describes this
                run only. When false, this run's counters and cycles add to
                the earlier runs'.
        """
        state = self.state
        if reset_stats:
            state.stats.reset()
        self._earlier = state.stats.total_cycles
        state.quit_requested = False
        state.halt = None
        self.cycles = 0
        self._own_limit = state.cycle_limit
        if self._limit is not None:
            state.cycle_limit = self._limit

    def admit(self) -> None:
        """Allow one more cycle, or fault F-9 at the PC it would run.

        Raises:
            EmulatorFault: F-9, the run has used ``state.cycle_limit`` cycles
                (a limit of 0 never faults). ``state.halt`` records it first.
        """
        limit = self.state.cycle_limit
        if limit and self.cycles >= limit:
            pc = self.state.program_counter
            self.state.halt = HaltRecord("fault", pc, FaultId.F9)
            raise EmulatorFault(FaultId.F9, pc, f"the run reached its cycle limit ({limit})")

    def count_cycle(self) -> None:
        """Count one completed cycle."""
        self.cycles += 1

    def end_run(self) -> None:
        """Write ``total_cycles`` and restore the state's own cycle limit;
        called when the run ends, faulted or not."""
        self.state.stats.total_cycles = self._earlier + self.cycles
        self.state.cycle_limit = self._own_limit
