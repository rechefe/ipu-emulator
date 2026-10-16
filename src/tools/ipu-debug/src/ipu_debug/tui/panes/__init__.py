"""One module per TUI pane, each implementing ``ipu_debug.tui.pane.Pane``.

- ``ipu_debug.tui.panes.xmem``: XMEM windows (E08-09);
- ``ipu_debug.tui.panes.registers``: the LR / CR scalar registers (E08-10).

A pane module imports no curses and knows nothing of the view: it renders
from a ``DebugSession`` and its own tabs, and mutates only through the session.
"""
