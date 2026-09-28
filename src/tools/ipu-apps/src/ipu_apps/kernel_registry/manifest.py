"""Every kernel, its cases and options, and the Bazel targets that run it.

    bazel run //src/tools/ipu-apps:kernel_manifest

prints one JSON document for tools (the VS Code extension), joining Bazel's
targets (``ipu_kernel_targets`` in ipu_app.bzl) with the registry's kernels,
cases and options (``case_options``, as the runner parses them). The two must
name the same kernels, so no target offered is missing; a kernel whose module
failed to import is listed under ``skipped`` instead.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path, PurePosixPath

from ipu_apps.kernel_registry import registry
from ipu_apps.kernel_registry.cases import case_options, load_cases


def _was_skipped(asm: str, skipped) -> bool:
    """Whether discovery skipped the kernel whose program is ``asm``: its package
    (its folder, from ``ipu_apps`` down) or one above it failed to import."""
    parts = PurePosixPath(asm).parent.parts
    package = ".".join(parts[parts.index("ipu_apps"):]) + "." if "ipu_apps" in parts else None
    return package is not None and any(
        package.startswith(m) or m.startswith(package) for m in (s.module + "." for s in skipped))


def build(targets: dict) -> dict:
    """The manifest, from Bazel's target file and the live registry."""
    discovered = registry.load()
    specs = {spec.name: spec for spec in registry.kernels()}
    declared = {
        name for name, bazel in targets["kernels"].items()
        if name in specs or not _was_skipped(bazel["asm"], discovered.skipped)
    }
    if declared != set(specs):
        raise ValueError(
            f"Bazel targets and the kernel registry disagree -- targets without a registered kernel: "
            f"{sorted(declared - set(specs)) or 'none'}; registered kernels without a target: {sorted(set(specs) - declared) or 'none'}")

    kernels, operations = {}, {}
    for name, spec in specs.items():
        bazel = targets["kernels"][name]
        operations.setdefault(spec.op, set()).update(spec.requires)
        kernels[name] = {
            "op": spec.op, "variant": spec.variant, "tags": list(spec.tags),
            "family": bazel["family"], "asm": bazel["asm"],
            "targets": {k: bazel[k] for k in ("run", "test", "benchmark") if k in bazel},
            # Loaded after discovery, not during it: case modules import numpy,
            # and discovery must stay cheap (test_harness_factory checks this).
            "cases": {
                case_name: {"options": case_options(case), "max_cycles": case.max_cycles}
                for case_name, case in load_cases(name).items()
            },
        }
    return {
        "kernels": kernels,
        "families": targets["families"],
        "operations": {op: {"params": sorted(params)} for op, params in operations.items()},
        "query": targets["query"],
        "skipped": [{"module": s.module, "error": s.error} for s in discovered.skipped],
    }


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: manifest.py <kernel_targets.json>", file=sys.stderr)
        return 2
    targets = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    try:
        manifest = build(targets)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    # What the paths are relative to, so a tool opened on a subfolder finds them.
    manifest["workspace"] = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
