"""Host-drivable, steppable IPU execution engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from ipu_emu.execute import (
    BreakResult,
    execute_instruction_skip_break,
    execute_next_instruction,
)
from ipu_emu.ipu_config import DEFAULT_DSTRUCTURE
from ipu_emu.regfile import RegFile
from ipu_emu.stats import RunStats
from ipu_emu.xmem import XMem
from ipu_emu.ipu_state import IpuState, INST_MEM_SIZE


class RunStatus(Enum):
    """Engine status after a step or run."""

    RUNNING = auto()
    HALTED = auto()
    BREAK = auto()


@dataclass
class EngineState:
    """Mutable execution-control state owned by the engine."""

    running: bool = False
    halted: bool = True
    break_hit: bool = False
    halt_requested: bool = False
    error: bool = False
    err_code: int = 0
    prog_len: int = INST_MEM_SIZE
    max_cycles: int = 100_000
    cycles: int = 0


class IpuEngine:
    """Steppable execution engine over an :class:`IpuState`."""

    def __init__(self, state: IpuState) -> None:
        self.state = state
        self.ctrl = EngineState()

    @property
    def running(self) -> bool:
        return self.ctrl.running

    @property
    def halted(self) -> bool:
        return self.ctrl.halted

    @property
    def break_hit(self) -> bool:
        return self.ctrl.break_hit

    @property
    def cycles(self) -> int:
        return self.ctrl.cycles

    @property
    def error(self) -> bool:
        return self.ctrl.error

    @property
    def err_code(self) -> int:
        return self.ctrl.err_code

    def set_prog_len(self, value: int) -> None:
        self.ctrl.prog_len = max(0, min(int(value), INST_MEM_SIZE))

    def set_max_cycles(self, value: int) -> None:
        self.ctrl.max_cycles = int(value) if value > 0 else 100_000

    def get_pc(self) -> int:
        return self.state.program_counter

    def set_pc(self, value: int) -> None:
        if self.ctrl.running:
            raise RuntimeError("PC write while running")
        pc = int(value)
        if pc < 0 or pc > INST_MEM_SIZE:
            raise ValueError(f"Invalid PC: {pc}")
        self.state.program_counter = pc

    def _is_program_halted(self) -> bool:
        return self.state.program_counter >= self.ctrl.prog_len

    def _sync_halt_status(self) -> RunStatus:
        if self._is_program_halted():
            self.ctrl.running = False
            self.ctrl.halted = True
            return RunStatus.HALTED
        if self.ctrl.break_hit:
            self.ctrl.running = False
            self.ctrl.halted = True
            return RunStatus.BREAK
        if self.ctrl.halt_requested:
            self.ctrl.running = False
            self.ctrl.halted = True
            self.ctrl.halt_requested = False
            return RunStatus.HALTED
        self.ctrl.running = True
        self.ctrl.halted = False
        return RunStatus.RUNNING

    def step(self, *, honor_break: bool = False) -> RunStatus:
        """Execute exactly one VLIW cycle."""
        if self._is_program_halted():
            self.ctrl.running = False
            self.ctrl.halted = True
            return RunStatus.HALTED

        limit = self.ctrl.max_cycles
        if self.ctrl.cycles >= limit:
            self.ctrl.running = False
            self.ctrl.halted = True
            return RunStatus.HALTED

        result = execute_next_instruction(self.state)
        if result == BreakResult.BREAK:
            if honor_break:
                self.ctrl.break_hit = True
                return self._sync_halt_status()
            execute_instruction_skip_break(self.state)

        self.ctrl.cycles += 1
        self.state.stats.total_cycles = self.ctrl.cycles
        return self._sync_halt_status()

    def run(self, max_cycles: int | None = None, *, honor_break: bool = False) -> RunStatus:
        """Run until halt, break, or cycle limit."""
        if max_cycles is not None:
            self.set_max_cycles(max_cycles)

        self.ctrl.running = True
        self.ctrl.halted = False
        self.ctrl.break_hit = False

        while True:
            status = self.step(honor_break=honor_break)
            if status != RunStatus.RUNNING:
                return status

    def start(self) -> RunStatus:
        """Start execution from the current PC (runs to completion)."""
        self.ctrl.break_hit = False
        self.ctrl.halt_requested = False
        return self.run()

    def halt(self) -> None:
        """Request a cooperative halt at the next instruction boundary."""
        self.ctrl.halt_requested = True

    def reset(self, *, preserve_imem: bool = True) -> None:
        """Soft reset. Preserves instruction memory when requested."""
        saved_inst = list(self.state.inst_mem) if preserve_imem else [None] * INST_MEM_SIZE
        prog_len = self.ctrl.prog_len if preserve_imem else INST_MEM_SIZE

        self.state.regfile = RegFile()
        self.state.set_cr_dstructure(
            valid_elements=DEFAULT_DSTRUCTURE.valid_elements,
            partition=DEFAULT_DSTRUCTURE.partition,
        )

        self.state.xmem = XMem()
        self.state.program_counter = 0
        self.state.stats = RunStats()
        self.state._debug_mult_stage_vectors.clear()
        self.state._debug_mult_stage_vectors_snap.clear()

        if preserve_imem:
            self.state.inst_mem = saved_inst
            self.ctrl.prog_len = prog_len

        self.ctrl = EngineState(prog_len=prog_len if preserve_imem else INST_MEM_SIZE)

    def clear_error(self) -> None:
        self.ctrl.error = False
        self.ctrl.err_code = 0

    def set_error(self, code: int) -> None:
        self.ctrl.error = True
        self.ctrl.err_code = int(code)
