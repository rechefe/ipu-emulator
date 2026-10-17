"""Compare an old kernel's state after setup with the new kernel's, before the run (E07-01).

    bazel run //kernels/tools:compare_setup -- OLD_KERNEL NEW_KERNEL [--case C]

OLD_KERNEL is an ``ipu_apps`` kernel name. NEW_KERNEL is a folder holding a
``kernel.toml`` (relative to where ``bazel run`` was started) or a kernel
name in the checkout's ``kernels/`` tree. ``--case`` names the old case (a
``CASES`` name or a ``k=v,...`` sweep id, default ``default``); the new case
of the same name runs, or the one ``--case-map OLD=NEW`` names.

The old harness is built through ``create_harness`` and runs only its
``setup()`` on a fresh state (FP32). The new kernel runs up to the point just
before its run: plan, place, pack, the input and scratch regions written and
the registers programmed (``ipu_harness.runner.stage``). Then every CR and
LR, and the XMEM bytes over the union of what either side wrote, are
compared. Each difference is named by register, or by the new placement's
region, row and lane.

Exits 0 when the two are identical, 1 when they differ, 2 on an error.
Temporary: deleted with ``ipu_apps`` in E07-25.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from migration import (
    SHOWN,
    case_format,
    legacy_case,
    legacy_setup,
    new_case,
    new_kernel,
    register_differences,
    written_ranges,
    xmem_differences,
)

from ipu_harness.runner import stage
from ipu_isa.errors import IpuError

__all__ = ["SetupReport", "compare_setup", "main"]


@dataclass(frozen=True)
class SetupReport:
    """How an old kernel's state after setup and a new kernel's staged state differ.

    Attributes:
        old: The old kernel and case, ``identity/default``.
        new: The new kernel and case.
        format: The format both ran in.
        registers: One line per CR or LR that differs.
        xmem: The first differing XMEM elements, located.
        xmem_total: How many XMEM elements differ.
        xmem_bytes: How many XMEM bytes were compared.
    """

    old: str
    new: str
    format: str
    registers: tuple[str, ...]
    xmem: tuple[str, ...]
    xmem_total: int
    xmem_bytes: int

    @property
    def identical(self) -> bool:
        return not self.registers and not self.xmem_total

    def describe(self) -> str:
        lines = [f"compare_setup: {self.old} (old) vs {self.new} (new), {self.format}"]
        if self.registers:
            lines.append(f"registers: {len(self.registers)} differ")
            lines.extend(f"  {line}" for line in self.registers)
        else:
            lines.append("registers: CR0-CR15 and LR0-LR15 identical")
        if self.xmem_total:
            lines.append(
                f"xmem: {self.xmem_total} element(s) differ in {self.xmem_bytes} bytes compared"
            )
            lines.extend(f"  {line}" for line in self.xmem)
            if self.xmem_total > SHOWN:
                lines.append(f"  ... and {self.xmem_total - SHOWN} more")
        else:
            lines.append(f"xmem: {self.xmem_bytes} bytes compared, identical")
        lines.append("IDENTICAL" if self.identical else "DIFFERENT")
        return "\n".join(lines)


def compare_setup(
    old_kernel: str,
    new: str,
    case_id: str = "default",
    *,
    case_map: dict[str, str] | None = None,
    root: Path | None = None,
) -> SetupReport:
    """Stage both kernels' case ``case_id`` and compare registers and XMEM.

    Args:
        old_kernel: The ``ipu_apps`` kernel.
        new: The new kernel's folder or name.
        case_id: The old case.
        case_map: Old case id -> new case name, where they differ.
        root: The kernel tree a name is looked up in.

    Returns:
        The report.
    """
    manifest = new_kernel(new, root)
    old_case = legacy_case(old_kernel, case_id)
    case = new_case(manifest, case_id, case_map or {})
    with tempfile.TemporaryDirectory(prefix="compare-setup-") as tmp:
        setup = legacy_setup(old_kernel, old_case, Path(tmp))
    staged = stage(manifest, case, case_format(setup.state, case))
    ranges = written_ranges(setup, staged)
    shown, total = xmem_differences(setup.state, staged, ranges)
    return SetupReport(
        old=f"{old_kernel}/{case_id}",
        new=f"{manifest.name}/{case.name}",
        format=staged.plan.format.value,
        registers=tuple(register_differences(setup.state, staged)),
        xmem=tuple(shown),
        xmem_total=total,
        xmem_bytes=sum(size for _, size in ranges),
    )


def case_mapping(pairs: Sequence[str]) -> dict[str, str]:
    """``--case-map OLD=NEW`` arguments as a mapping."""
    mapping = {}
    for pair in pairs:
        old, sep, new = pair.partition("=")
        if not sep or not old or not new:
            raise argparse.ArgumentTypeError(f"--case-map takes OLD=NEW, not {pair!r}")
        mapping[old] = new
    return mapping


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("old", help="the ipu_apps kernel")
    parser.add_argument("new", help="the new kernel's folder (kernel.toml) or name")
    parser.add_argument("--case", default="default", help="the old case (default: default)")
    parser.add_argument("--case-map", action="append", default=[], metavar="OLD=NEW")
    parser.add_argument("--kernels-root", type=Path, help="the kernel tree (default: kernels/)")
    args = parser.parse_args(argv)
    try:
        report = compare_setup(
            args.old,
            args.new,
            args.case,
            case_map=case_mapping(args.case_map),
            root=args.kernels_root,
        )
    except (IpuError, ValueError, OSError, argparse.ArgumentTypeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(report.describe())
    return 0 if report.identical else 1


if __name__ == "__main__":
    sys.exit(main())
