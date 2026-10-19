"""The IPU emulator's interactive debugger: a curses TUI (``ipu_debug.tui``) over
``ipu_emu.emulator.run_with_debug``.

The TUI is an adapter over ``ipu_debug.session.DebugSession`` (parsing,
register and XMEM reads, disassembly; no curses, no I/O) and lays values out
with the display helpers of ``ipu_debug.format``. No module imports another's
private names.

Depends one way on ``ipu_emu`` and ``ipu_isa``; the emulator does not import it.
"""
