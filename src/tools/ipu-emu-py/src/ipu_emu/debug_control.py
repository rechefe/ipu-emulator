"""Execution controls shared by debugger frontends, independent of curses."""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field

from ipu_emu.ipu_state import INST_MEM_SIZE, IpuState


def validate_pc(pc: int) -> int:
    if not 0 <= pc < INST_MEM_SIZE:
        raise ValueError(f"PC must be between 0 and {INST_MEM_SIZE - 1}")
    return pc


@dataclass
class DebugControl:
    """Controls retained for the lifetime of one IPU state.

    Every change to the breakpoints or to a user-chosen PC goes through a
    method here, which validates it; ``breakpoints`` is a read-only view.
    """

    _breakpoints: set[int] = field(default_factory=set, init=False)
    run_target: int | None = None
    stop_reason: str = "paused"
    kernel_name: str | None = None
    last_completed_pc: int | None = None
    last_completed_instruction: dict[str, int] | None = None

    @property
    def breakpoints(self) -> frozenset[int]:
        return frozenset(self._breakpoints)

    def add_breakpoint(self, pc: int) -> None:
        self._breakpoints.add(validate_pc(pc))

    def remove_breakpoint(self, pc: int) -> None:
        """Remove the breakpoint at *pc*; nothing happens if there is none."""
        self._breakpoints.discard(validate_pc(pc))

    def clear_breakpoints(self) -> None:
        self._breakpoints.clear()

    def toggle_breakpoint(self, pc: int) -> bool:
        validate_pc(pc)
        if pc in self._breakpoints:
            self._breakpoints.remove(pc)
            return False
        self._breakpoints.add(pc)
        return True

    @staticmethod
    def set_pc(state: IpuState, pc: int) -> None:
        """Move *state* to a user-chosen *pc*, which must be in instruction memory."""
        state.program_counter = validate_pc(pc)

    def run_until(self, pc: int, current_pc: int) -> bool:
        """Arm a one-stop target; return whether execution should resume."""
        validate_pc(pc)
        self.run_target = pc if pc != current_pc else None
        return pc != current_pc


_CONTROLS: weakref.WeakKeyDictionary[IpuState, DebugControl] = weakref.WeakKeyDictionary()


def get_debug_control(state: IpuState) -> DebugControl:
    control = _CONTROLS.get(state)
    if control is None:
        control = DebugControl()
        _CONTROLS[state] = control
    return control
