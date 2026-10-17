"""Prove a migrated family output-identical to its old kernels (E07-01, D3).

    bazel run //kernels/tools:compare_legacy -- FAMILY [--kernel NAME ...]

For each old ``ipu_apps`` kernel of FAMILY (or each ``--kernel``) and the new
kernel of the same name under the checkout's ``kernels/<FAMILY>/``:

* binary: the new default case's program (what ``assemble_<name>`` builds)
  has the E02 asm digest of the old ``.asm``, and each case's program is the
  old harness's program for that case, byte for byte;
* output: the old harness's run still has its E02 output digest (a check on
  the baseline itself), and the new run's output regions
  (``RunResult.raw_outputs``) hold the same bytes as the old state's XMEM
  at those rows. Comparing regions, not files, makes the check independent of
  how each old ``teardown`` wrote its file;
* cycles: the same on both sides.

A multi-stage kernel (``[kernel] stages``, E06-14) is compared the same way:
its programs are the distinct programs its stages assembled, in order, held
to the old harness's distinct programs, and the E02 digest must be one of
them; its cycles are its stages' sum. Distinct programs do not show the order
the stages ran them in (A, B, A against A, A, B): only the output check
catches a reordering.

Every case of the old kernel's E02 baseline runs, in FP32, and the new case
of the same name (or ``--case-map OLD=NEW``) runs on the same input values:
the migration keeps the old seeds and generators (the per-family checklist,
step 4). The report is a Markdown table for the PR description.

Exits 0 when every row passes, 1 otherwise.
Temporary: deleted with ``ipu_apps`` in E07-25.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from compare_setup import case_mapping
from migration import (
    case_format,
    e02_asm_digest,
    family_of,
    kernels_root,
    legacy_case,
    legacy_case_ids,
    legacy_run,
    new_case,
    sha256,
)

from ipu_apps.kernel_registry import baseline
from ipu_apps.kernel_registry.registry import kernel_spec, kernels
from ipu_harness.assembly import AssemblyCache
from ipu_harness.cases import Case
from ipu_harness.manifest import KernelManifest
from ipu_harness.placement import Placement
from ipu_harness.registry import Registry
from ipu_harness.runner import RunResult, run_kernel, stage
from ipu_isa.errors import IpuError
from ipu_isa.formats import FORMATS, ElementFormat

__all__ = ["Row", "compare_kernel", "main", "table"]


@dataclass(frozen=True)
class Row:
    """One case of one kernel, old against new.

    Attributes:
        kernel: The kernel.
        case: The old case id.
        binary_ok: The programs are identical (and the E02 digest holds).
        output_ok: The output regions are identical (and the E02 output digest holds).
        cycles_old: The old run's cycles, if it ran.
        cycles_new: The new run's cycles, if it ran.
        notes: Why a check failed.
    """

    kernel: str
    case: str
    binary_ok: bool
    output_ok: bool
    cycles_old: int | None
    cycles_new: int | None
    notes: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return (
            self.binary_ok
            and self.output_ok
            and self.cycles_old is not None
            and self.cycles_old == self.cycles_new
        )


def _mark(ok: bool) -> str:
    return "yes" if ok else "**NO**"


def table(rows: Sequence[Row]) -> str:
    """The rows as a Markdown table, then one line per failed check."""
    lines = [
        "| kernel | case | binary ok | output ok | cycles old/new |",
        "|--------|------|-----------|-----------|----------------|",
    ]
    for row in rows:
        old = "-" if row.cycles_old is None else str(row.cycles_old)
        new = "-" if row.cycles_new is None else str(row.cycles_new)
        cycles = f"{old}/{new}" if row.cycles_old == row.cycles_new else f"**{old}/{new}**"
        lines.append(
            f"| {row.kernel} | {row.case} | {_mark(row.binary_ok)} | {_mark(row.output_ok)} "
            f"| {cycles} |"
        )
    passed = sum(row.passed for row in rows)
    lines.append("")
    lines.append(f"{passed} of {len(rows)} case(s) identical.")
    for row in rows:
        lines.extend(f"- {row.kernel}/{row.case}: {note}" for note in row.notes)
    return "\n".join(lines)


def _default_digests(manifest: KernelManifest, cache: AssemblyCache) -> set[str]:
    """The digests of the new default case's programs in its first format:
    ``assemble_<name>``'s, or a multi-stage kernel's stages'."""
    default = new_case(manifest, "default", {})
    fmt = default.formats[0]
    if manifest.stages is None:
        return {sha256(stage(manifest, default, fmt, cache=cache).program.to_bin())}
    return {sha256(program) for program in _stage_programs(manifest, default, fmt)[1]}


def _stage_programs(
    manifest: KernelManifest, case: Case, fmt: ElementFormat
) -> tuple[RunResult, list[bytes]]:
    """A multi-stage run, and the distinct programs its stages assembled, in order."""
    own = AssemblyCache()  # this run's programs only
    result = run_kernel(manifest, case, fmt, cache=own)
    return result, [program.to_bin() for program in own.programs]


def _new_run(
    manifest: KernelManifest, case: Case, fmt: ElementFormat, cache: AssemblyCache
) -> tuple[RunResult, list[bytes], Placement]:
    """The new kernel's run of ``case``, its programs, and where its tensors were."""
    if manifest.stages is None:
        staged = stage(manifest, case, fmt, cache=cache)
        return (
            run_kernel(manifest, case, fmt, cache=cache),
            [staged.program.to_bin()],
            staged.placement,
        )
    new, programs = _stage_programs(manifest, case, fmt)
    if new.placement is None:
        raise ValueError(f"{manifest.name}/{case.name}: the run reports no placement")
    return new, programs, new.placement


def _distinct(programs: Sequence[bytes]) -> list[bytes]:
    """``programs`` without repeats, in first-run order (as ``AssemblyCache.programs``)."""
    return list(dict.fromkeys(programs))


def compare_kernel(
    name: str, manifest: KernelManifest | None, case_map: dict[str, str] | None = None
) -> list[Row]:
    """Every case of the old kernel ``name`` against the new ``manifest``.

    Args:
        name: The ``ipu_apps`` kernel.
        manifest: The new kernel; None if it does not exist yet (one failed row).
        case_map: Old case id -> new case name, where they differ.

    Returns:
        One row per case.
    """
    if manifest is None:
        return [Row(name, "-", False, False, None, None, ("no new kernel of this name",))]
    spec = kernel_spec(name)
    cache = AssemblyCache()
    e02 = e02_asm_digest(spec)
    kernel_notes = []
    try:
        digests: set[str] | None = _default_digests(manifest, cache)
    except (IpuError, ValueError) as error:
        digests = None
        kernel_notes.append(f"the new default case does not assemble: {error}")
    e02_ok = e02 is not None and digests is not None and e02 in digests
    if e02 is None:
        kernel_notes.append("the old program has no E02 asm digest")
    elif digests is not None and not e02_ok:
        shown = ", ".join(sorted(d[:12] for d in digests))
        if manifest.stages is None:
            kernel_notes.append(f"assemble_{name} digest {shown} is not the E02 {e02[:12]}")
        else:
            kernel_notes.append(f"no stage program's digest ({shown}) is the E02 {e02[:12]}")
    kernel = _Kernel(name, manifest, case_map or {}, cache, e02_ok, tuple(kernel_notes))
    return [_compare_case(kernel, case_id) for case_id in legacy_case_ids(name)]


@dataclass(frozen=True)
class _Kernel:
    """What every case of one kernel shares."""

    name: str
    manifest: KernelManifest
    case_map: dict[str, str]
    cache: AssemblyCache
    e02_ok: bool  #: the new default program has the old program's E02 digest
    notes: tuple[str, ...]


def _compare_case(kernel: _Kernel, case_id: str) -> Row:
    name, manifest, cache = kernel.name, kernel.manifest, kernel.cache
    notes = list(kernel.notes)
    try:
        case = new_case(manifest, case_id, kernel.case_map)
        with tempfile.TemporaryDirectory(prefix="compare-legacy-") as tmp:
            old = legacy_run(name, legacy_case(name, case_id), Path(tmp))
        fmt = case_format(old.state, case)
        new, programs, placement = _new_run(manifest, case, fmt, cache)
    except (IpuError, ValueError, OSError, AssertionError, RuntimeError) as error:
        return Row(name, case_id, False, False, None, None, (*notes, f"{error}"))

    old_programs = _distinct(old.programs)
    binary_ok = kernel.e02_ok and old_programs == programs
    if len(old_programs) != len(programs):
        notes.append(
            f"the old harness ran {len(old_programs)} distinct program(s); the new kernel "
            f"{len(programs)}"
        )
    elif old_programs != programs:
        notes.append("the case's programs differ from the old harness's")

    output_ok = True
    want = baseline.load(name).get(case_id)
    if want is None:
        output_ok = False
        notes.append("the old case has no E02 baseline record")
    elif want.output_sha256 != old.record.output_sha256:
        output_ok = False
        notes.append("the old output no longer has its E02 output digest")
    if new.fault is not None:
        output_ok = False
        notes.append(f"the new run faulted: {new.fault.describe()}")
    row_bytes = FORMATS[fmt].row_bytes
    for output, data in new.raw_outputs.items():
        region = placement.region(output)
        start, size = region.base_row * row_bytes, region.rows * row_bytes
        if old.state.xmem.snapshot_bytes(start, size) != data:
            output_ok = False
            notes.append(f"output {output} differs from the old XMEM at bytes {start}..")
    if old.cycles != new.cycles:
        notes.append(f"cycles differ: old {old.cycles}, new {new.cycles}")
    return Row(name, case_id, binary_ok, output_ok, old.cycles, new.cycles, tuple(notes))


def compare_family(
    family: str,
    *,
    only: Sequence[str] = (),
    case_map: dict[str, str] | None = None,
    root: Path | None = None,
) -> list[Row]:
    """Every old kernel of ``family`` (or those in ``only``) against the new tree."""
    names = sorted(spec.name for spec in kernels() if family_of(spec) == family)
    if only:
        unknown = sorted(set(only) - set(names))
        if unknown:
            raise ValueError(f"no old {family} kernel {', '.join(unknown)} (kernels: {names})")
        names = [n for n in names if n in only]
    if not names:
        raise ValueError(f"no old kernels in family {family!r}")
    tree = kernels_root(root)
    registry = Registry.load(tree, families=[family]) if (tree / family).is_dir() else None
    rows = []
    for name in names:
        manifest = None
        if registry is not None and name in registry.manifests:
            manifest = registry.kernel(name)
        elif registry is not None:
            for toml, error in registry.invalid.items():
                if toml.parent.name == name:
                    raise error
        rows.extend(compare_kernel(name, manifest, case_map))
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("family", help="the kernel family, e.g. projections")
    parser.add_argument("--kernel", action="append", default=[], help="only this old kernel")
    parser.add_argument("--case-map", action="append", default=[], metavar="OLD=NEW")
    parser.add_argument("--kernels-root", type=Path, help="the kernel tree (default: kernels/)")
    args = parser.parse_args(argv)
    try:
        rows = compare_family(
            args.family,
            only=args.kernel,
            case_map=case_mapping(args.case_map),
            root=args.kernels_root,
        )
    except (IpuError, ValueError, OSError, argparse.ArgumentTypeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(table(rows))
    return 0 if all(row.passed for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
