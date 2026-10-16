"""Direct TUI launch for applications selected by ``bazel run --config=debug``."""

from __future__ import annotations

import sys

from ipu_emu.debug_control import DebugControl
from ipu_emu.emulator import DebugAction, DebugCallback


def make_tui_debug_callback(kernel_name: str | None = None) -> tuple[DebugCallback, DebugControl]:
    """The curses debugger as a ``run_with_debug`` callback, with the control
    to pass to that run (``run_test(..., debug_control=...)``); the callback
    keeps the control and the view between stops. Cancellation exits before
    application output handling."""
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise RuntimeError("IPU debugging requires an interactive input and output terminal")

    from ipu_emu import debug_tui

    control = DebugControl(kernel_name=kernel_name)
    session = None

    def callback(state, cycle):
        nonlocal session
        if session is None:
            session = debug_tui.new_debug_view_session(state)
        # Call curses directly so initialization failures never fall through
        # into an unattended run or the deprecated line debugger.
        action = debug_tui.run_debug_tui(state, cycle, control=control, session=session)
        if action == DebugAction.QUIT:
            # Unwinds runner TemporaryDirectory contexts, but does not return
            # incomplete state for teardown, output comparisons, or PASS reports.
            raise SystemExit(0)
        return action

    return callback, control
