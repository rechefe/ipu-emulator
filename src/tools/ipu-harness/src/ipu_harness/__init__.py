"""The kernel framework: everything between a kernel's declaration and a checked run.

A kernel is its assembly plus a ``kernel.toml`` declaration (ADR 0006). This
package reads the declaration, validates parameters once, places regions in
XMEM, packs data per element format, programs CRs, assembles, runs on the
emulator and compares against a reference. Each of those is one module, added
by the E06 issue that needs it: ``manifest``, ``params``, ``placement``,
``packing``, ``abi``, ``assembly``, ``cases``, ``reference``, ``tolerance``,
``runner``, ``stages``, ``registry`` and ``testing``.

It imports ``ipu_isa``, the assembler's session API (``ipu_as``) and the
emulator's reference-model API (``ipu_emu``). ``ipu_kernels`` and ``torch``
only inside the functions that need them: kernel-side modules are loaded by
name when asked for, and torch is a test-only dependency.
``tools/import_graph.toml`` enforces the package-level rules.
"""
