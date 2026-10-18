"""The curses TUI: a view over ``DebugSession``, drawn through a ``Screen``.

- ``ipu_debug.tui.view``: ``CursesDebugView`` (layout, focus, status line,
  help and the main loop) and ``run_debug_tui``, the entry point a debugger
  callback calls at each stop;
- ``ipu_debug.tui.workspace``: ``DebugViewSession``, what persists between stops;
- ``ipu_debug.tui.pane``: the ``Pane`` interface each pane implements;
- ``ipu_debug.tui.panes``: one module per pane (pipeline, disassembly, XMEM,
  registers);
- ``ipu_debug.tui.layout``: the splits and the rectangle of each pane;
- ``ipu_debug.tui.chrome``: frames, header, tab bars, footer and scroll thumbs;
- ``ipu_debug.tui.pointer``: clicks, the wheel and drags;
- ``ipu_debug.tui.help``: the help overlay;
- ``ipu_debug.tui.prompt``: the open editor's prompt overlay;
- ``ipu_debug.tui.draw``: glyphs, styles and the line canvas panes draw on;
- ``ipu_debug.tui.editor``: the tab, value and scalar editors panes open;
- ``ipu_debug.tui.tabs``: ``TabSet``, one pane's tabs;
- ``ipu_debug.tui.keymap``: every key, its command and its help text;
- ``ipu_debug.tui.screen``: the terminal seam, the only module importing
  ``curses`` (``CursesScreen``, and ``FakeScreen`` for tests).
"""

from ipu_debug.tui.panes.disassembly import DisassemblyTab
from ipu_debug.tui.panes.pipeline import PipelineTab
from ipu_debug.tui.panes.registers import RegisterTab
from ipu_debug.tui.panes.xmem import XmemTab
from ipu_debug.tui.view import CursesDebugView, run_debug_tui
from ipu_debug.tui.workspace import DebugViewSession, new_debug_view_session

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
