# One mutation path: register set, breakpoints and run-until in `DebugSession`

Part of [Epic: Debugger, generators and editor](00-epic.md) · ID E08-05 · Size M · Depends on: E08-04 · Behaviour change: yes (CLI `set` now also invalidates cached XMEM views, and the TUI register editor accepts `pc`, matching each other)

## Goal

Every state change a user can make from a debugger exists once, in `DebugSession`. Both frontends call it. The two diverged copies of "set a register" merge, and views learn about changes through a generation counter instead of clearing each other's caches.

## Background

- Register set exists twice:
  - `cli.do_set` (old `debug_cli.py:948-980`) handles `pc` and rejects byte-array registers, but does not clear XMEM caches;
  - `tui._submit_scalar_editor` (old `debug_tui.py:3200-3218`) clears `_xmem_requests` and `_xmem_cache`, but has no `pc` path.
- Breakpoint add, toggle and delete and run-until are called from both frontends with slightly different error handling (CLI: `_command_pc` writes the error; TUI: status line).

## Implementation

1. Add to `DebugSession`:

   ```python
   generation: int                                          # bumped on every mutation
   def set_register(self, name: str, value: int) -> RegisterView   # "pc" included; DebugError on bad name, dtype or range
   def add_breakpoint(self, pc: int) -> None
   def remove_breakpoint(self, pc: int) -> None
   def clear_breakpoints(self) -> None
   def toggle_breakpoint(self, pc: int) -> bool
   def run_until(self, pc: int) -> bool                     # True when execution should resume
   ```

   Every method validates through `RunControl` (E08-01) and raises `DebugError` with the text the frontends show today.
2. CLI `do_set`, `do_break`, `do_delete` and `do_until` become two-line adapters: call the session, then print the result or the `DebugError`.
3. The TUI editor, breakpoint keys and run-to-cursor call the same methods. XMEM, register and pipeline views cache by `session.generation` and recompute when it changes. Delete the explicit `_xmem_cache.clear()` calls.

## Tests

- [ ] Adapter conformance: for a table of commands (`set lr3 5`, `set pc 4`, `break 7`, `delete all`, `until 9`), running them through `DebuggerCLI.onecmd` and calling `DebugSession` directly leave identical state.
- [ ] A TUI test on a fake screen: editing a CR invalidates the XMEM view (generation changed), and the editor accepts `pc`.
- [ ] Every other existing debugger test passes.
- [ ] `bazel test //...` green.

## Acceptance Criteria

- [ ] `grep -n "set_scalar\|program_counter =" src/tools/ipu-debug/src/ipu_debug/{cli,tui}.py` is empty. Only `session.py` mutates state.
- [ ] No frontend clears another component's cache.
