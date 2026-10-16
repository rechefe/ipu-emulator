"""The IPU emulator's interactive debugger: a line CLI (``ipu_debug.cli``) and a
curses TUI (``ipu_debug.tui``) over ``ipu_emu.emulator.run_with_debug``.

Both frontends are adapters over ``ipu_debug.session.DebugSession`` (parsing,
register and XMEM reads, disassembly; no curses, no I/O) and share display
helpers from ``ipu_debug.format``. Neither imports the other's private names.

Depends one way on ``ipu_emu`` and ``ipu_isa``; the emulator does not import it.
"""
