"""The legacy kernel runner: per-kernel run code and its registry, which
``ipu_harness`` and ``ipu_kernels`` replace in E07.

Each kernel under ``ipu_apps.kernels`` is a Bazel target that runs its cases
on the emulator; ``ipu_apps.kernel_registry`` routes a shape query to one.
"""
