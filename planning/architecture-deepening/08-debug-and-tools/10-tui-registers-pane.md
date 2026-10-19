# TUI registers pane module

Part of [Epic: Debugger, generators and editor](00-epic.md) · ID E08-10 · Size M · Depends on: E08-08 · Behaviour change: none

## Goal

Move the registers pane into `ipu_debug/tui/panes/registers.py`: register groups, cell layout, change highlighting, and the scalar editor. It reads only `DebugSession.registers()` and mutates only through `DebugSession.set_register` (E08-05).

## Background

- `_draw_registers` (old `debug_tui.py:1564-1673`) and `_draw_register_cell` (old `:1674-1774`) draw it.
- The editor entry and submit live elsewhere in the class (`_submit_scalar_editor`, old `:3200`).
- The pane read raw bytes with `RegFile.raw()` (old `:621`) before E08-04.

## Implementation

1. Create `panes/registers.py` with `RegistersPane.render` and `RegistersPane.handle`. Group definitions (`all`, `lr`, `cr`, vector registers) come from the register schema in `ipu_isa`, not from a hard-coded list.
2. The editor is a `Command.EDIT_VALUE` handled by the pane. It calls `session.set_register` and shows `DebugError` text on the status line.
3. Delete the moved methods from `CursesDebugView`.

## Tests

- [ ] E08-08 golden screen tests for register sequences are unchanged.
- [ ] Pure `render` tests: every register dtype, change highlighting after a step, a narrow width.
- [ ] `bazel test //...` green.

## Acceptance Criteria

- [ ] No register drawing or editing logic remains in `CursesDebugView`.
- [ ] No register name list is hard-coded in the pane.
