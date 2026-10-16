"""Kernel-side Python shared across kernels: references, pack transforms, family code.

The kernels themselves live under the top-level ``kernels/`` tree and need no
Python package. What several of them share lives here, in one subpackage per
concern:

* ``ipu_kernels.reference``: shared references, one module per operation.
* ``ipu_kernels.pack``: named pack transforms, each over one declared tensor.
* ``ipu_kernels.<family>``: family code that is neither a kernel nor a
  framework concern.

It builds on ``ipu_harness``. Nothing imports it except kernel tests and the
registry loader (``tools/import_graph.toml``).
"""
