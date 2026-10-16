"""ipu_kernel: the Bazel macro that declares one kernel (E06).

    load("//kernels:defs.bzl", "ipu_kernel")

A stub until E06-13 fills it in (generated kernel tests); calling it fails, so
nothing can depend on a macro that does not exist yet.
"""

def ipu_kernel(name, **kwargs):
    """Declare kernel `name` from its .asm, kernel.toml and optional reference.py.

    Args:
        name: the kernel's name, the last path component of //kernels/<family>/<name>.
        **kwargs: reserved for E06-13.
    """
    fail("ipu_kernel(%s): not implemented until E06-13" % name)
