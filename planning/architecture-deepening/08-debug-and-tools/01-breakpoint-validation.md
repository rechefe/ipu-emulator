# Validate every breakpoint and PC mutation in `DebugControl`

Part of [Epic: Debugger, generators and editor](00-epic.md) · ID E08-01 · Size S · Depends on: — · Behaviour change: yes (CLI `set pc N` with N outside the instruction memory reports an error instead of writing it)

## Goal

Every change to breakpoints or the PC from a debugger frontend goes through a validated `DebugControl` method. The breakpoint set cannot be mutated from outside. This small fix lands before the debugger is restructured.

## Background

- `DebugControl` (`ipu_emu/debug_control.py:18-43`) validates in `add_breakpoint`, `toggle_breakpoint` and `run_until` (via `validate_pc`, `:11-14`), but exposes `breakpoints` as a plain mutable `set`.
- `DebuggerCLI.do_delete` bypasses it: `self.control.breakpoints.clear()` (`debug_cli.py:867`) and `.discard(pc)` (`:872`). The PC is validated by `_command_pc` (`:836-844`), but the set is mutated directly.
- `do_set` writes `self.state.program_counter = value` with no range check (`debug_cli.py:960-961`). A PC beyond `INST_MEM_SIZE` then fails later, far from the cause. `do_quit` legitimately sets `INST_MEM_SIZE` as the halt sentinel (`debug_cli.py:819`), but that is internal, not user input.

## Implementation

1. `DebugControl`:
   - `breakpoints` becomes a private `_breakpoints: set[int]`, exposed as a read-only `frozenset` property;
   - add `remove_breakpoint(pc)` (validated; a no-op if absent) and `clear_breakpoints()`;
   - add `set_pc(state, pc)`, which validates and then writes.
2. `do_delete` uses `remove_breakpoint` / `clear_breakpoints`.
3. `do_set pc N` uses `DebugControl.set_pc`. It prints the `validate_pc` error message for an out-of-range value. `do_quit` keeps writing the sentinel directly.
4. Search the TUI for direct `breakpoints.` mutations and route them through the methods (`grep -n "breakpoints\." debug_tui.py`).

## Tests

- [ ] `control.breakpoints.add(3)` raises `AttributeError` (a frozenset has no `add`).
- [ ] `delete 5`, `delete all` and `break 5` behave as before (`test_debug_cli` unchanged).
- [ ] `set pc 99999` prints `PC must be between 0 and …` and leaves the PC unchanged.
- [ ] `bazel test //src/tools/ipu-emu-py/...` green.

## Acceptance Criteria

- [ ] No code outside `debug_control.py` mutates the breakpoint set.
- [ ] A user-supplied PC is always validated.
