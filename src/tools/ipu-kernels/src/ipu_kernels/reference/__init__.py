"""Shared references by operation: what a kernel's output should be, computed independently.

A reference never imports the emulator (``ipu_emu``): one that does is a copy
of the code under test, not a reference. This subpackage's own row in
``tools/import_graph.toml`` allows only ``ipu_isa``, so it reaches the
emulator neither directly nor through ``ipu_harness`` or the rest of
``ipu_kernels``.
"""
