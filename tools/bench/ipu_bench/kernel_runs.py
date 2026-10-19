"""Kernel-tree cases and programs for the benchmark modules (E07-25).

Each benchmark runs a kernel of ``kernels/`` through the harness runner, as
``bazel test //kernels/...`` runs it, and checks the outputs; only the
emulated run (``model.run``) is timed.
"""

from __future__ import annotations

import time


def run_timed(kernel: str, case_name: str, fmt: str | None = None, repeats: int = 1) -> dict:
    """Run one case ``repeats`` times; the fastest emulated run and its cycles.

    Args:
        kernel: A kernel or registry entry of the kernel tree, by name.
        case_name: One of its cases.
        fmt: The element format (``ElementFormat`` value); ``None``: the
            case's first.
        repeats: How many runs; the program is assembled once.

    Returns:
        ``{"cycles", "seconds"}``: the run's cycles and its best time.

    Raises:
        RuntimeError: A run did not pass (a fault, or an output off its golden).
    """
    from ipu_harness.assembly import AssemblyCache
    from ipu_harness.cases import expand
    from ipu_harness.registry import default_registry
    from ipu_harness.runner import RunOptions, run_kernel
    from ipu_isa.formats import ElementFormat

    manifest = default_registry().kernel(kernel)
    case = next(c for c in expand(manifest) if c.name == case_name)
    element = case.formats[0] if fmt is None else ElementFormat(fmt)
    cache = AssemblyCache()
    best = None
    cycles = 0
    for _ in range(max(1, repeats)):
        seconds = []

        def timed(context, seconds=seconds):
            start = time.perf_counter()
            context.model.run(context.cycle_limit)
            seconds.append(time.perf_counter() - start)

        result = run_kernel(manifest, case, element, RunOptions(executor=timed), cache=cache)
        if not result.passed:
            raise RuntimeError(result.describe())
        cycles = result.cycles
        best = seconds[0] if best is None else min(best, seconds[0])
    return {"cycles": cycles, "seconds": best}


def has_case(kernel: str, case_name: str, fmt: str | None = None) -> bool:
    """Whether the kernel tree has this kernel, case and format."""
    from ipu_harness.cases import expand
    from ipu_harness.registry import default_registry

    manifest = default_registry().kernel(kernel)
    return any(
        c.name == case_name and (fmt is None or fmt in {f.value for f in c.formats})
        for c in expand(manifest)
    )


def default_programs() -> list[tuple[str, str, int]]:
    """Each kernel's default program, rendered: ``(name, text, words)``.

    The program ``:assemble_<name>`` writes (its first format), for every
    kernel and template of the tree, sorted by name.
    """
    import warnings

    from ipu_harness.kernel_bin import default_programs as programs_of
    from ipu_harness.registry import Registry, kernel_tree

    registry = Registry.load(kernel_tree())
    manifests = sorted(
        [*registry.manifests.values(), *registry.templates.values()], key=lambda m: m.name
    )
    found = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*-only .* slot")
        for manifest in manifests:
            program = next(iter(programs_of(manifest).values()))
            found.append((manifest.name, program.rendered, len(program.words)))
    return found
