# TUI disassembly and pipeline pane modules

Part of [Epic: Debugger, generators and editor](00-epic.md) · ID E08-11 · Size M · Depends on: E08-08, E08-07 · Behaviour change: none

## Goal

Move the last two panes into `ipu_debug/tui/panes/disassembly.py` and `ipu_debug/tui/panes/pipeline.py`. After this, `CursesDebugView` is only the shell: layout, focus, status line, help and the main loop.

## Background

- `_draw_disassembly` (old `debug_tui.py:1367-1536`), `_draw_stages` (old `:1537-1563`) and `_draw_pipeline_register` (old `:1960-2172`) draw these panes.
- Breakpoint toggling and run-to-cursor are key handlers in the class.
- The pipeline data comes from `pipeline.stage_operations`, which E08-07 rebuilds on the decoded word.

## Implementation

1. `DisassemblyPane`:
   - renders `session.disassemble(start, count)`;
   - marks the current PC and breakpoints;
   - handles `TOGGLE_BREAKPOINT` and `RUN_TO_CURSOR` through `DebugSession`.
2. `PipelinePane` renders the stage view and the pipeline-register tabs from E08-07's `stage_operations` and from `session.registers()`.
3. Delete the moved methods. `CursesDebugView` keeps only layout, focus, status and help, and calls the panes.

## Tests

- [ ] E08-08 golden screen tests are unchanged.
- [ ] Pure `render` tests for both panes.
- [ ] Measure `test_debug_tui.py`: fewer than 30 `monkeypatch` uses and no private-name calls. Move tests to per-pane test files.
- [ ] `bazel test //...` green.

## Acceptance Criteria

- [ ] `CursesDebugView` (or its successor `TuiApp`) is under 600 lines. No function in `ipu_debug/tui/` is longer than about 80 lines.
