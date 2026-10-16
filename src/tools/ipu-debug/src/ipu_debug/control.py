"""Execution controls shared by debugger frontends, independent of curses."""

from __future__ import annotations

from dataclasses import dataclass

from ipu_emu.ipu_state import INST_MEM_SIZE, IpuState
from ipu_emu.run_control import RunControl


def validate_pc(pc: int) -> int:
    if not 0 <= pc < INST_MEM_SIZE:
        raise ValueError(f"PC must be between 0 and {INST_MEM_SIZE - 1}")
    return pc


@dataclass
class DebugControl(RunControl):
    """A debug run's controls as its frontends change them.

    Every change to the breakpoints or to a user-chosen PC goes through a
    method here, which validates it; ``breakpoints`` is a read-only view.
    Whoever starts the run creates one and gives it to both the run
    (``run_with_debug(..., control=...)``) and its frontends.
    """

    kernel_name: str | None = None

    def add_breakpoint(self, pc: int) -> None:
        self._breakpoints = self._breakpoints | {validate_pc(pc)}

    def remove_breakpoint(self, pc: int) -> None:
        """Remove the breakpoint at *pc*; nothing happens if there is none."""
        self._breakpoints = self._breakpoints - {validate_pc(pc)}

    def clear_breakpoints(self) -> None:
        self._breakpoints = frozenset()

    def toggle_breakpoint(self, pc: int) -> bool:
        validate_pc(pc)
        enabled = pc not in self._breakpoints
        self._breakpoints = self._breakpoints ^ {pc}
        return enabled

    @staticmethod
    def set_pc(state: IpuState, pc: int) -> None:
        """Move *state* to a user-chosen *pc*, which must be in instruction memory."""
        state.program_counter = validate_pc(pc)

    def run_until(self, pc: int, current_pc: int) -> bool:
        """Arm a one-stop target; return whether execution should resume."""
        validate_pc(pc)
        self.run_target = pc if pc != current_pc else None
        return pc != current_pc
