# TUI XMEM pane module

Part of [Epic: Debugger, generators and editor](00-epic.md) · ID E08-09 · Size M · Depends on: E08-08 · Behaviour change: none

## Goal

Move everything about the XMEM pane into `ipu_debug/tui/panes/xmem.py`, implementing the `Pane` interface from E08-08:

- tab targets (`row|byte BASE OFFSET COUNT FORMAT`);
- cell formatting (hex, i8, u8, f32, 16-bit cells);
- scrolling and markers;
- the add and edit tab editor.

`render` is pure: session and view state in, lines out.

## Background

- `_draw_xmem` (old `debug_tui.py:1775-1915`, 140 lines) is the longest draw function. It mixes request resolution, the cache, formatting and curses calls.
- XMEM requests are resolved through `DebugSession.parse_xmem` / `read_xmem` (E08-04), and the cache is keyed by `session.generation` (E08-05).
- Related helpers are spread across the class: `_clean_xmem_marker` (old `:1916`), the request parsing near old `:1334-1344` and `:3099`.

## Implementation

1. Create `panes/xmem.py` with `XmemPane.render` and `XmemPane.handle`, plus small pure helpers (`format_row(cells, fmt, width) -> Line`). No function longer than about 60 lines (Q66).
2. Move the XMEM tab editor's parse and validate logic into `handle` (`Command.EDIT_TAB`, `Command.ADD_TAB`).
3. Delete the moved methods from `CursesDebugView`.

## Tests

- [ ] E08-08 golden screen tests for XMEM sequences are unchanged.
- [ ] New pure tests for `XmemPane.render` cover every format, a window crossing a row boundary, a narrow terminal, and a stale generation recomputing. They use no `FakeScreen` and no monkeypatch.
- [ ] `bazel test //...` green.

## Acceptance Criteria

- [ ] No XMEM logic remains in `CursesDebugView`.
- [ ] `XmemPane` has no curses import.
