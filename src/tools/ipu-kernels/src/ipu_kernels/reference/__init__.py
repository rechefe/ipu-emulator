"""Shared references by operation: what a kernel's output should be, computed independently.

A reference never imports the emulator (``ipu_emu``): one that does is a copy
of the code under test, not a reference. ``tools/import_graph.toml`` enforces
it with this subpackage's own row.
"""
