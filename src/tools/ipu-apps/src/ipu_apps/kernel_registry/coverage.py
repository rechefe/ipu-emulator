"""Coverage reports derived from the kernels themselves.

Every figure here is obtained by *asking* kernels what they support, never by
reading a maintained description. A hand-written coverage table is a second
copy of the routing rules, and the second copy is the one that goes stale --
this codebase had exactly that: an error message advising "use softmax_columns
for width >= 128" that survived the boundary moving to 65.

:func:`boundaries` probes a parameter across a range and reports where the
winning kernel changes, which is how the routing tables in the documentation
are produced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from ipu_apps.kernel_registry.registry import kernels, load, operations, resolve


@dataclass(frozen=True)
class Boundary:
    """A contiguous run of values handled by one kernel (or by none)."""

    start: int
    end: int
    kernel: str | None

    def render(self, param: str) -> str:
        span = f"{param} = {self.start}" if self.start == self.end else (
            f"{param} {self.start}..{self.end}"
        )
        return f"{span:<28} {self.kernel or '(not covered)'}"


def boundaries(
    op: str,
    param: str,
    values: Iterable[int],
    *,
    package: str = "ipu_apps",
    build: Callable[[int], Any] | None = None,
    **fixed: Any,
) -> tuple[Boundary, ...]:
    """Sweep an integer parameter over ``values`` and collapse runs of the same
    winning kernel.

    This is what makes a routing table trustworthy: it reports what the
    registry *actually answers*, so it cannot describe behaviour the kernels do
    not have.

    Args:
        op:      Operation to probe.
        param:   Name of the parameter being swept.
        values:  Integer values to try. Runs are collapsed on these, so they
            should be consecutive for the ranges to read naturally.
        build:   Optional ``int -> value`` mapping the swept integer onto the
            parameter's real type. Needed when the interesting axis is inside a
            structured parameter -- e.g. sweeping the row length of a shape:
            ``boundaries("softmax", "shape", range(120, 136),
            build=lambda n: (8, n), dim=1)``.
        **fixed: Other parameters held constant across the sweep.
    """
    runs: list[Boundary] = []
    for value in values:
        arg = build(value) if build else value
        verdict = resolve(op, package=package, **{param: arg}, **fixed)
        winner = verdict.app_name if verdict.supported else None
        if runs and runs[-1].kernel == winner and runs[-1].end == value - 1:
            runs[-1] = Boundary(runs[-1].start, value, winner)
        else:
            runs.append(Boundary(value, value, winner))
    return tuple(runs)


def report(
    *,
    package: str = "ipu_apps",
    probes: Sequence[tuple[str, str, Iterable[int], dict[str, Any]]] = (),
) -> str:
    """Render the full coverage report.

    Args:
        package: Root package to inspect.
        probes:  Optional ``(op, param, values, fixed)`` sweeps to render as
            routing tables. Without probes the report still lists every kernel,
            operation and discovery failure.
    """
    found = load(package)
    lines: list[str] = ["# Kernel coverage", ""]

    ops = operations(package=package)
    lines.append(f"Operations: {', '.join(ops) if ops else '(none)'}")
    lines.append(f"Kernels:    {len(found.specs)}")
    lines.append("")

    for op in ops:
        lines.append(f"## {op}")
        lines.append("")
        for spec in kernels(op, package=package):
            variant = f" [{spec.variant}]" if spec.variant else ""
            tags = f"  tags: {', '.join(spec.tags)}" if spec.tags else ""
            lines.append(f"- **{spec.name}**{variant} -- `{spec.app_class.__module__}`{tags}")
        lines.append("")

    for op, param, values, fixed in probes:
        fixed = dict(fixed)
        build = fixed.pop("build", None)
        label = fixed.pop("label", param)
        fixed_note = f" (with {', '.join(f'{k}={v}' for k, v in fixed.items())})" if fixed else ""
        lines.append(f"### routing: {op} by `{label}`{fixed_note}")
        lines.append("")
        lines.append("```")
        for b in boundaries(op, param, values, package=package, build=build, **fixed):
            lines.append(b.render(label))
        lines.append("```")
        lines.append("")

    if found.skipped:
        # Surfaced, not swallowed: a module that failed to import is a kernel
        # potentially missing from coverage, which is exactly the kind of hole
        # a coverage report exists to reveal.
        lines.append("## Modules skipped during discovery")
        lines.append("")
        for s in found.skipped:
            lines.append(f"- `{s.module}` -- {s.error}")
        lines.append("")

    return "\n".join(lines)
