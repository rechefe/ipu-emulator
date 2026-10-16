"""The curses TUI: a view over ``DebugSession``, drawn through a ``Screen``.

- ``ipu_debug.tui.view``: ``CursesDebugView`` (layout, frame, overlays) and
  ``run_debug_tui``, the entry point a debugger callback calls at each stop;
- ``ipu_debug.tui.pane``: the ``Pane`` interface each pane implements;
- ``ipu_debug.tui.panes``: one module per pane (XMEM, registers);
- ``ipu_debug.tui.draw``: glyphs, styles and the line canvas panes draw on;
- ``ipu_debug.tui.editor``: the tab, value and scalar editors panes open;
- ``ipu_debug.tui.tabs``: ``TabSet``, one pane's tabs;
- ``ipu_debug.tui.keymap``: every key, its command and its help text;
- ``ipu_debug.tui.screen``: the terminal seam, the only module importing
  ``curses`` (``CursesScreen``, and ``FakeScreen`` for tests).
"""

from ipu_debug.tui.panes.registers import RegisterTab
from ipu_debug.tui.panes.xmem import XmemTab
from ipu_debug.tui.view import (
    CursesDebugView,
    DebugViewSession,
    DisassemblyTab,
    PipelineTab,
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
