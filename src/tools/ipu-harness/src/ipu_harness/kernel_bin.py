"""A kernel's default-case program as a ``.bin`` file: the ``:assemble_<name>`` target.

    kernel_bin.py KERNEL OUT.bin

``ipu_kernel()`` and ``ipu_kernel_family()`` (kernels/defs.bzl) declare
``:assemble_<name>``, which writes ``<name>.bin``: the program rendered with the
default case's values in its first format (``--format`` for another) and
assembled, the binary E07 compares with the E02 asm digest of the old kernel
(E06-07, E06-17). KERNEL is its ``kernel.toml`` (or folder), or
``<registry.toml>::<name>`` for a family entry.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from ipu_as.session import AssemblyResult
from ipu_harness.assembly import assemble
from ipu_harness.cases import expand
from ipu_harness.family import load_kernel
from ipu_harness.manifest import KernelManifest
from ipu_harness.params import plan_case
from ipu_isa.errors import IpuError
from ipu_isa.formats import ElementFormat


def default_programs(manifest: KernelManifest) -> dict[ElementFormat, AssemblyResult]:
    """The default case's program in each format the case runs in.

    Args:
        manifest: The kernel (its ``asm`` may be any file, such as a docs listing).

    Returns:
        Each format's assembly, in the case's format order.

    Raises:
        ConfigurationError: The default case does not plan.
        ipu_as.errors.AssemblerError: The program does not assemble.
    """
    # The default case's first point, should it sweep.
    case = next(c for c in expand(manifest) if c.decl is manifest.cases["default"])
    return {
        fmt: assemble(plan_case(manifest, case.values, fmt, case=case.decl)) for fmt in case.formats
    }


def default_bin(kernel: str) -> bytes:
    """The ``.bin`` bytes of a kernel's default case in its first format.

    Args:
        kernel: A kernel reference (:func:`ipu_harness.family.split_ref`).

    Returns:
        The assembled program, as ``ipu-as`` writes a ``.bin``.

    Raises:
        ConfigurationError: The kernel does not load or plan.
        ipu_as.errors.AssemblerError: Its program does not assemble.
    """
    return next(iter(default_programs(load_kernel(kernel)).values())).to_bin()


def main(argv: Sequence[str] | None = None) -> int:
    """The command line. Returns the exit code."""
    parser = argparse.ArgumentParser(prog="kernel_bin.py", description=__doc__.splitlines()[0])
    parser.add_argument("kernel", help="kernel.toml, a kernel folder, or REGISTRY.toml::NAME")
    parser.add_argument("output", type=Path, help="the .bin file to write")
    args = parser.parse_args(argv)
    try:
        args.output.write_bytes(default_bin(args.kernel))
    except (IpuError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
