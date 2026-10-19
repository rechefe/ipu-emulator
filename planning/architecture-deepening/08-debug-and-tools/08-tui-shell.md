# TUI shell: `TabSet`, screen adapter and keymap

Part of [Epic: Debugger, generators and editor](00-epic.md) · ID E08-08 · Size M · Depends on: E08-05 · Behaviour change: none

## Goal

Prepare the curses TUI to be split into one module per pane (Q67). This issue builds the frame the panes plug into:

- a generic `TabSet` replaces four parallel tab-state triples;
- a thin `Screen` adapter is the only code that touches curses;
- key handling becomes a keymap table.

After this, each pane can move out in its own PR (E08-09 to E08-11).

## Background

- `DebugViewSession` (old `debug_tui.py:475-500`) keeps four parallel triples:
  - `tabs` / `active_tab` / `tab_offset`;
  - `disassembly_tabs` / `active_disassembly_tab` / `disassembly_tab_offset`;
  - `register_tabs` / …;
  - `pipeline_tabs` / ….

  Helpers such as `_pane_tabs(pane)` and `_active_pane_tab_index(pane)` switch on the pane name.
- `CursesDebugView` (old `debug_tui.py:779-3703`) has 130 methods. It calls curses directly in many places, including `curses.curs_set` inside `_clear_editor` with an `except Exception: pass`.
- `handle_key` (old `:2539-2683`) is a 145-line if/elif chain.
- `test_debug_tui.py` (3139 lines) uses `monkeypatch` 153 times, several of them on curses internals.

## Implementation

1. `ipu_debug/tui/tabs.py`:

   ```python
   @dataclass
   class TabSet(Generic[T]):
       tabs: list[T]; active: int = 0; offset: int = 0
       def add(self, tab: T) -> None; def close(self, index: int) -> None
       def select(self, index: int) -> None; def current(self) -> T | None
   ```

   `DebugViewSession` holds `panes: dict[PaneId, TabSet]`. Delete the twelve fields and the per-pane switch helpers.
2. `ipu_debug/tui/screen.py`:
   - a `Screen` protocol with `size()`, `draw(y, x, text, style)`, `clear()`, `refresh()`, `read_key()` and `cursor(visible)`;
   - a `CursesScreen` adapter, the only module importing `curses`;
   - a `FakeScreen` for tests, which records draws into a character grid.

   There are two adapters, so the seam is real.
3. `ipu_debug/tui/keymap.py`: `KEYMAP: dict[tuple[Focus, Key], Command]`. `handle_key` looks up the command and dispatches to a small method. Help text (`_draw_help`, old `:2340`) is generated from the keymap, so the two cannot drift.
4. Define the pane interface the next issues implement:

   ```python
   class Pane(Protocol):
       id: PaneId
       def render(self, session: DebugSession, view: PaneState, rect: Rect) -> list[Line]   # pure
       def handle(self, command: Command, session: DebugSession, view: PaneState) -> Outcome
   ```

   Existing drawing code is wrapped to satisfy it for now. The real split happens per pane.

## Tests

- [ ] Golden screen tests: for a fixed program and a fixed key sequence, the `FakeScreen` grid equals the grid recorded from the base commit, through a recording `Screen` patched into the old code once. Cover 10 representative sequences: open tab, close tab, switch focus, edit register, scroll XMEM, step, continue, toggle breakpoint, resize, help.
- [ ] `TabSet` unit tests.
- [ ] The keymap help test: every keymap entry appears in the help overlay.
- [ ] `bazel test //...` green.

## Acceptance Criteria

- [ ] Only `screen.py` imports `curses`.
- [ ] `handle_key` is under 30 lines.
- [ ] The golden screen tests exist and are the proof for E08-09 to E08-11.
