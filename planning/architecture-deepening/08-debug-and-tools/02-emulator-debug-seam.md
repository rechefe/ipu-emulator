# Explicit debug seam in the emulator: no `WeakKeyDictionary`, no debugger re-exports

Part of [Epic: Debugger, generators and editor](00-epic.md) · ID E08-02 · Size S · Depends on: E08-01 · Behaviour change: none

## Goal

The emulator offers the debugger one explicit seam: `run_with_debug(state, callback, control=…)`. It knows nothing else about debugger frontends. This is the prerequisite for moving the debugger into its own package (E08-03) with a one-way dependency `ipu_debug → ipu_emu` (Q62).

## Background

- `debug_control.py:46-54` keeps a module-global `WeakKeyDictionary[IpuState, DebugControl]`. `get_debug_control(state)` is called by:
  - `emulator.run_with_debug` (`emulator.py:23, 115`);
  - `debug_launch.py:19-20`, which writes `kernel_name` into it;
  - the TUI (`debug_tui.py:43`).

  Hidden state is attached to `IpuState` through a side channel.
- `run_with_debug` (`emulator.py:89-162`) reads `control.breakpoints` and `control.run_target`, and writes `stop_reason`, `last_completed_pc` and `last_completed_instruction`.
- `ipu_emu/__init__.py:25-32` re-exports `debug_prompt`, `DebugCLI`, `print_all_registers`, `state_to_json_dict`, `save_state_json` and `disassemble_current` from `debug_cli`. Importing the emulator therefore imports the debugger.
- D5: only the reference-model API (E04) is stable, so these names may change freely.

## Implementation

1. Split `DebugControl` into the part the run loop needs. It stays in `ipu_emu` as `ipu_emu.run_control.RunControl`:
   - `breakpoints` (read-only), `run_target`;
   - the out-fields `stop_reason`, `last_completed_pc` and `last_completed_instruction`.

   `kernel_name` moves to the frontend side (E08-06).
2. Change the signature to `run_with_debug(state, debug_callback, max_cycles, *, control: RunControl | None = None, break_on_entry=False)`. With `None`, it creates a fresh `RunControl`. The caller that wants to share it with a frontend creates it and passes it.
3. Thread the control object explicitly: `run_test(..., debug_control=…)` → `run_with_debug`. The debugger frontends receive it from their creator (`debug_launch` builds it and closes over it).
4. Delete `_CONTROLS` and `get_debug_control`. There are about 5 call sites, so no deprecation shim is needed (D5).
5. Remove the `debug_cli` re-exports from `ipu_emu/__init__.py`. Update importers (`grep -rn "from ipu_emu import" src | grep -i debug`) to import from the debugger module directly.
6. If E04 has already landed its stepping engine or `step()` API, express `run_with_debug` over it. Otherwise keep the loop and leave a note for E04.

## Tests

- [ ] `test_debug_cli`, `test_debug_tui` and `ipu-apps:test_debug_launch` pass. Use `test_debug_tui`'s existing stop-reason assertions as the behaviour proof.
- [ ] A new test runs two `IpuState`s with separate controls in one process. Breakpoints on one do not affect the other.
- [ ] `python -c "import ipu_emu"` does not import `ipu_emu.debug_cli` (checked via `sys.modules`).
- [ ] `bazel test //...` green.

## Acceptance Criteria

- [ ] `grep -rn "WeakKeyDictionary\|get_debug_control" src` is empty.
- [ ] `ipu_emu/__init__.py` imports no `debug_*` module.
