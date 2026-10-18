"""High-level emulator runner: load a program and run it.

Provides two modes:
  * ``run_until_complete`` — runs the program silently, ignoring break-
    points, until a HALT word retires or a word faults. Fetching past the
    loaded program faults F-2; reaching the cycle limit faults F-9.
  * ``run_with_debug``   — honours breakpoints.  On break, invokes a
    user-supplied callback that decides whether to step, continue, or
    quit.

Both functions operate on an :class:`IpuState` that has already been
loaded with an assembled program (via :func:`load_program`). Each counts its
cycles with a :class:`~ipu_emu.run_clock.RunClock`: ``state.stats`` starts from
zero (unless ``reset_stats=False``), ``total_cycles`` is set even when the run
raises, and the run faults F-9 at ``state.cycle_limit`` cycles.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum, auto

from ipu_emu.cycle import CycleOutcome
from ipu_emu.errors import ConfigurationError, EmulatorFault
from ipu_emu.ipu import Ipu
from ipu_emu.ipu_state import DEFAULT_CYCLE_LIMIT, INST_MEM_SIZE, IpuState
from ipu_emu.run_clock import RunClock
from ipu_emu.run_control import RunControl


class DebugAction(Enum):
    """Action returned by a debug callback on breakpoint."""

    CONTINUE = auto()
    STEP = auto()
    QUIT = auto()


# Type alias for the debug callback.
# Receives (state, cycle_count) and returns a DebugAction.
DebugCallback = Callable[[IpuState, int], DebugAction]

# The cycle limit every run entry point defaults to (``IpuState``'s default).
# Kernels that need more pass their own; 0 disables the check.
DEFAULT_MAX_CYCLES = DEFAULT_CYCLE_LIMIT


def _word_at(state: IpuState, pc: int):
    """The IMEM word at *pc*, or ``None`` outside IMEM (the fetch faults F-2)."""
    return state.inst_mem[pc] if 0 <= pc < len(state.inst_mem) else None


def load_program(state: IpuState, instructions: list[dict[str, int]]) -> None:
    """Load decoded instruction dicts into the state's instruction memory.

    Sets ``state.program_length``: fetching past the last word is fault F-2.
    """
    if len(instructions) > INST_MEM_SIZE:
        raise ConfigurationError(
            f"Program too large: {len(instructions)} instructions (max {INST_MEM_SIZE})"
        )
    for i, inst in enumerate(instructions):
        state.inst_mem[i] = inst
    state.program_length = len(instructions)


def run_until_complete(
    state: IpuState,
    max_cycles: int | None = None,
    *,
    raise_on_fault: bool = True,
    reset_stats: bool = True,
) -> int:
    """Run the program ignoring breakpoints until it halts.

    Returns the number of cycles executed. ``state.stats`` describes this run
    only, unless *reset_stats* is false (then it accumulates); its
    ``total_cycles`` is set even when the run raises. ``state.halt`` says how
    the run ended. *max_cycles*, when given, is ``state.cycle_limit`` for this
    run (0: no limit); else the state's own limit holds.

    Raises ``EmulatorFault`` when a word faults (the faulting word is not
    counted), including F-9 at the cycle limit, unless *raise_on_fault* is
    false: the run then returns, and ``state.halt`` holds the fault.
    """
    # One engine for the whole run: it carries no per-cycle state beyond the
    # snapshot buffer it reuses, and building one per cycle was pure overhead.
    ipu = Ipu(state)
    cycle = ipu.cycle  # works while ``ipu`` is alive, which it is for the run
    clock = RunClock(state, max_cycles)
    clock.begin_run(reset_stats=reset_stats)
    try:
        # Until a HALT or a fault: a PC past the program, IMEM included, is
        # fetched and faults F-2 (``is_halted`` also counts a PC past IMEM).
        while state.halt is None:
            clock.admit()
            # A BREAK does not stop a normal run: the word retires.
            if cycle.step() is CycleOutcome.BREAK:
                cycle.retire()
            clock.count_cycle()
    except EmulatorFault:
        if raise_on_fault:
            raise
    finally:
        clock.end_run()
    return clock.cycles


def run_with_debug(
    state: IpuState,
    debug_callback: DebugCallback,
    max_cycles: int | None = None,
    *,
    control: RunControl | None = None,
    break_on_entry: bool = False,
    reset_stats: bool = True,
) -> int:
    """Run the program honouring breakpoints.

    When a breakpoint fires, *debug_callback(state, cycle_count)* is called.
    The callback returns a :class:`DebugAction`:

    - ``CONTINUE`` — resume execution until the next embedded or runtime
      breakpoint, or a temporary run target.
    - ``STEP`` — execute the current instruction, then break again.
    - ``QUIT`` — stop the run here and set ``state.quit_requested``; the
      program is left unfinished.

    Stops occur before instruction side effects. Callback cycle counts and
    the return value count completed instructions. *control* holds the
    breakpoints and run target the run honours, and receives its stop reason
    and last completed instruction; a caller that shares it with a debugger
    frontend passes it in, else a fresh one is used.
    ``break_on_entry`` stops at the prepared PC before the first instruction.

    ``state.stats`` describes this run only, unless *reset_stats* is false;
    its ``total_cycles`` is set even when the run raises. *max_cycles*, when
    given, is ``state.cycle_limit`` for this run (0: no limit); reaching the
    limit faults F-9.
    """
    stepping = False
    ipu = Ipu(state)
    cycle = ipu.cycle
    clock = RunClock(state, max_cycles)
    clock.begin_run(reset_stats=reset_stats)

    # TODO: express this loop over the steppable engine
    # (planning/riscv-host-integration/03-steppable-engine.md).
    control = control if control is not None else RunControl()
    control.last_completed_pc = None
    control.last_completed_instruction = None
    entry_pending = break_on_entry
    try:
        # As in run_until_complete; a debugger's quit (PC past IMEM) ends the
        # run at the callback below.
        while state.halt is None:
            clock.admit()
            issued_pc = state.program_counter
            issued_instruction = _word_at(state, issued_pc)
            reasons = []
            if entry_pending:
                reasons.append("entry")
                entry_pending = False
            if state.program_counter in control.breakpoints:
                reasons.append("breakpoint")
            if state.program_counter == control.run_target:
                reasons.append("run target")
            if stepping:
                reasons.append("step")
            # A requested stop must not execute the instruction. Still check
            # its BREAK slot so simultaneous causes produce one callback.
            result = cycle.evaluate_break() if reasons else cycle.step()
            if result == CycleOutcome.BREAK:
                reasons.append("BREAK instruction")
            if reasons:
                control.stop_reason = ", ".join(reasons)
                control.run_target = None
                action = debug_callback(state, clock.cycles)
                if action == DebugAction.QUIT or state.is_halted:
                    state.quit_requested = action == DebugAction.QUIT
                    break
                stepping = action == DebugAction.STEP
                # Resume the paused instruction exactly once, including when
                # the callback edits PC. A later loop visit can stop again.
                issued_pc = state.program_counter
                issued_instruction = _word_at(state, issued_pc)
                cycle.retire()
            control.last_completed_pc = issued_pc
            control.last_completed_instruction = (
                dict(issued_instruction) if issued_instruction is not None else None
            )
            clock.count_cycle()
    finally:
        control.run_target = None
        clock.end_run()
    return clock.cycles
