"""The kernel framework: everything between a kernel's declaration and a checked run.

A kernel is its assembly plus a ``kernel.toml`` declaration (ADR 0006). This
package reads the declaration, validates parameters once, places regions in
XMEM, packs data per element format, programs CRs, assembles, runs on the
emulator and compares against a reference. The pipeline, in order:
``manifest`` (with ``expressions`` and ``conditions``), ``params``,
``placement``, ``packing``, ``abi``, ``assembly``, ``cases``, ``reference``,
``runner`` (with ``stages`` and ``chain``) and ``compare`` (with
``tolerance``). Around it: ``family`` (a family's variants), ``registry``
and ``query`` (which kernel computes an op), ``cli``, ``debug``,
``benchmark``, ``inspector``, and the modules behind each ``ipu_kernel()``
target (``kernel_test``, ``kernel_cases``, ``kernel_fixtures``,
``kernel_bin``, ``kernel_page``, ``manifest_json``, ``registry_bzl``,
``rendered_programs``).

It imports ``ipu_isa``, the assembler's session API (``ipu_as``) and the
emulator's reference-model API (``ipu_emu``). Never ``ipu_kernels``: kernel-side
modules are loaded by a name read from registry data, as plugins. ``torch``
only inside the functions that need it, because it is a test-only dependency.
``tools/import_graph.toml`` enforces the package-level rules.
"""
