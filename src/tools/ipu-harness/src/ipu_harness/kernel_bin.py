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

from ipu_harness.assembly import assemble
from ipu_harness.cases import expand
from ipu_harness.family import load_kernel
from ipu_harness.params import plan_case
from ipu_isa.errors import IpuError


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
    manifest = load_kernel(kernel)
    # The default case's first point, should it sweep.
    case = next(c for c in expand(manifest) if c.decl is manifest.cases["default"])
    plan = plan_case(manifest, case.values, case.formats[0], case=case.decl)
    return assemble(plan).to_bin()


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
