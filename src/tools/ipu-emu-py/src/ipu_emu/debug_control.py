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
    """Controls retained for the lifetime of one IPU state."""

    breakpoints: set[int] = field(default_factory=set)
    run_target: int | None = None
    stop_reason: str = "paused"

    def add_breakpoint(self, pc: int) -> None:
        self.breakpoints.add(validate_pc(pc))

    def toggle_breakpoint(self, pc: int) -> bool:
        validate_pc(pc)
        if pc in self.breakpoints:
            self.breakpoints.remove(pc)
            return False
        self.breakpoints.add(pc)
        return True

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
