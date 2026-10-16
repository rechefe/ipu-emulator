"""The curses TUI: a view over ``DebugSession``, drawn through a ``Screen``.

- ``ipu_debug.tui.view``: ``CursesDebugView`` (layout, frame, overlays) and
  ``run_debug_tui``, the entry point a debugger callback calls at each stop;
- ``ipu_debug.tui.pane``: the ``Pane`` interface each pane implements;
- ``ipu_debug.tui.tabs``: ``TabSet``, one pane's tabs;
- ``ipu_debug.tui.keymap``: every key, its command and its help text;
- ``ipu_debug.tui.screen``: the terminal seam, the only module importing
  ``curses`` (``CursesScreen``, and ``FakeScreen`` for tests).
"""

from ipu_debug.tui.view import (
    CursesDebugView,
    DebugViewSession,
    DisassemblyTab,
    PipelineTab,
    RegisterTab,
    XmemTab,
    new_debug_view_session,
    run_debug_tui,
)

__all__ = [
    "CursesDebugView",
    "DebugViewSession",
    "DisassemblyTab",
    "PipelineTab",
    "RegisterTab",
    "XmemTab",
    "new_debug_view_session",
    "run_debug_tui",
]
