# TUI error hygiene: no `except Exception: pass`

Part of [Epic: Debugger, generators and editor](00-epic.md) · ID E08-12 · Size S · Depends on: E08-09, E08-10, E08-11 · Behaviour change: yes (an unexpected exception is logged and shown on the status line instead of being silently swallowed)

## Goal

The TUI catches only what it expects: `curses.error` from terminal limits, and `DebugError` for user mistakes. Anything else is a bug that must be visible.

## Background

- `except Exception: pass` (old `debug_tui.py`):
  - `:2536` in the draw path;
  - `:3262` in `_clear_editor` around `curses.curs_set`;
  - `:3700` in the main loop.
- `run_debug_tui` (old `:3729`) catches `Exception` broadly.
- `AttributeError` probes (old `:887-957`, for example `:945`) test for curses capabilities by catching exceptions.

## Implementation

1. Replace each `except Exception: pass`:
   - with `except curses.error` inside `CursesScreen` (E08-08), where terminal-size failures are expected;
   - with `except DebugError as e: status(e)` in command handling.
2. Replace the `AttributeError` probes with explicit capability checks in `CursesScreen.__init__` (`hasattr(curses, "…")`, `curses.has_colors()`), stored as flags.
3. The top-level handler in `run_debug_tui` restores the terminal, logs the traceback to a file (`$TMPDIR/ipu-debug-<pid>.log`), prints the path after curses ends, and re-raises.

## Tests

- [ ] A pane that raises `RuntimeError` in `render` is shown on the status line and logged. The test asserts the log file exists and the exception propagates after cleanup.
- [ ] `curses.error` on a too-small terminal is still tolerated (golden screen test at 20x5).
- [ ] `bazel test //...` green.

## Acceptance Criteria

- [ ] `grep -rn "except Exception" src/tools/ipu-debug` matches only the top-level handler, which re-raises.
