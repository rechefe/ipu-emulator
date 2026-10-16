"""Direct TUI launch for applications selected by ``bazel run --config=debug``."""

from __future__ import annotations

import sys

from ipu_debug.control import DebugControl
from ipu_emu.emulator import DebugCallback


def make_tui_debug_callback(kernel_name: str | None = None) -> tuple[DebugCallback, DebugControl]:
    """The curses debugger as a ``run_with_debug`` callback, with the control
    to pass to that run (``run_test(..., debug_control=...)``).

    The callback builds one ``DebugSession`` (named *kernel_name*) and one
    view at the first stop and keeps them between stops. It returns the
    user's action; on ``DebugAction.QUIT`` the run stops with
    ``state.quit_requested`` set, and the caller decides what that means.
    """
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise RuntimeError("IPU debugging requires an interactive input and output terminal")

    from ipu_debug import tui as debug_tui
    from ipu_debug.session import DebugSession

    control = DebugControl()
    debug = None
    view = None

    def callback(state, cycle):
        nonlocal debug, view
        if debug is None:
            debug = DebugSession(state, control, kernel_name=kernel_name)
            view = debug_tui.new_debug_view_session(state)
        # Call curses directly so initialization failures never fall through
        # into an unattended run or the deprecated line debugger.
        return debug_tui.run_debug_tui(state, cycle, control=control, session=view, debug=debug)

    return callback, control
