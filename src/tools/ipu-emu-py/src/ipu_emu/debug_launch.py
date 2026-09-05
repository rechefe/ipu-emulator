"""Direct TUI launch for applications selected by ``bazel run --config=debug``."""

from __future__ import annotations

import sys

from ipu_emu.emulator import DebugAction, DebugCallback


def make_tui_debug_callback() -> DebugCallback:
    """Start in curses; cancellation exits before application output handling."""
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise RuntimeError("IPU debugging requires an interactive input and output terminal")

    from ipu_emu.debug_tui import run_debug_tui

    def callback(state, cycle):
        # Call curses directly so initialization failures never fall through
        # into an unattended run or the deprecated line debugger.
        action = run_debug_tui(state, cycle)
        if action == DebugAction.QUIT:
            # Unwinds runner TemporaryDirectory contexts, but does not return
            # incomplete state for teardown, output comparisons, or PASS reports.
            raise SystemExit(0)
        return action

    return callback
