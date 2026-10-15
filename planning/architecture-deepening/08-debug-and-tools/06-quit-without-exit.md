# QUIT without `SystemExit`; kernel name passed explicitly

Part of [Epic: Debugger, generators and editor](00-epic.md) · ID E08-06 · Size S · Depends on: E08-04 · Behaviour change: none (the user sees the same result: quitting ends the run with no PASS line)

## Goal

Quitting the debugger is a return value, not a process exit raised from inside the emulator's callback. Only the CLI entry point turns it into an exit code. The kernel name the TUI shows comes from the session's constructor, not from a side channel.

## Background

- `launch.py` (old `debug_launch.py:17-28`):
  - writes `kernel_name` into the global control (`:19-20`);
  - raises `SystemExit(0)` inside the callback on `DebugAction.QUIT` (`:23-26`), so teardown, output comparison and the `PASS` line are skipped by unwinding the whole process.
- `kernel_registry/base.py:118-123` also raises `SystemExit(0)` on `KeyboardInterrupt` in debug mode.
- E06-12 makes the runner own the debug TUI launch and alias profiling. This issue gives the runner the signal it needs.

## Implementation

1. `make_tui_debug_callback(kernel_name)` builds `DebugSession(state, control, kernel_name=kernel_name)` lazily on the first stop and keeps it in the closure. Delete the side-channel write.
2. On QUIT the callback returns `DebugAction.QUIT`. `run_with_debug` already breaks out on it (`emulator.py:148-149`). `run_test` returns a `RunOutcome` (or sets `state.quit_requested`). The runner checks it, skips teardown, compare and `PASS`, and exits 0 from its `main`.
3. The `KeyboardInterrupt` path in `base.py` returns the same "quit" outcome instead of raising `SystemExit`. E06-12 (the runner owns the debug TUI launch) moves this into the runner; coordinate there.

## Tests

- [ ] Runner-level test (headless, with a callback that returns QUIT at the first stop): the run ends, `TemporaryDirectory` contexts are cleaned up, no `PASS` line is printed, the exit code is 0, and no `SystemExit` is raised below `main`.
- [ ] `ipu-apps:test_debug_launch` passes.
- [ ] `bazel test //...` green.

## Acceptance Criteria

- [ ] `grep -rn "SystemExit" src/tools/ipu-debug src/tools/ipu-emu*` is empty.
