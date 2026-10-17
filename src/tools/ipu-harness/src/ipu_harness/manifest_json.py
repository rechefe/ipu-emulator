"""Every kernel of the kernel tree, its cases and its targets, as JSON (E06-13).

    bazel run //kernels:manifest

joins the targets file ``ipu_kernel_tree()`` writes (kernels/defs.bzl) with the
routing registry (E06-16) and prints one JSON document for tools, with the same
keys as ``//src/tools/ipu-apps:kernel_manifest``. The two must name the same
kernels, so no target offered is missing; a ``kernel.toml`` that fails to load
is listed under ``skipped``.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ipu_emu.errors import ConfigurationError
from ipu_harness.cases import expand
from ipu_harness.registry import Registry, kernel_tree


def build(targets: Mapping[str, Any], registry: Registry) -> dict[str, Any]:
    """The manifest, from the targets file and the registry.

    Raises:
        ConfigurationError: The targets and the registry name different kernels.
    """
    declared = set(targets["kernels"])
    skipped_dirs = {path.parent.name for path in registry.invalid}
    loaded = set(registry.manifests)
    if declared - skipped_dirs != loaded:
        raise ConfigurationError(
            "kernels/BUILD.bazel and the kernel tree disagree -- targets without a kernel: "
            f"{sorted(declared - skipped_dirs - loaded) or 'none'}; kernels without a target: "
            f"{sorted(loaded - declared) or 'none'}"
        )
    kernels: dict[str, Any] = {}
    operations: dict[str, set[str]] = {}
    for name, manifest in sorted(registry.manifests.items()):
        bazel = targets["kernels"][name]
        operations.setdefault(manifest.op, set()).update(manifest.params)
        kernels[name] = {
            "op": manifest.op,
            "variant": name,
            "tags": [],
            "family": bazel["family"],
            "asm": f"{bazel['dir']}/{manifest.asm.name}",
            "targets": {k: bazel[k] for k in ("run", "test", "nightly")},
            "formats": [f.value for f in manifest.formats],
            "cases": {
                case.name: {"options": dict(case.values), "max_cycles": None, "tier": case.tier}
                for case in expand(manifest)
            },
        }
    families = sorted({kernel["family"] for kernel in kernels.values()})
    return {
        "kernels": kernels,
        "families": {family: {} for family in families},
        "operations": {op: {"params": sorted(params)} for op, params in operations.items()},
        "query": targets["query"],
        "skipped": [
            {"module": str(path), "error": str(error)} for path, error in registry.invalid.items()
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Print the manifest. Returns the exit code."""
    args = sys.argv[1:] if argv is None else list(argv)
    if len(args) != 1:
        print("usage: manifest_json.py <kernel_targets.json>", file=sys.stderr)
        return 2
    targets = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    try:
        manifest = build(targets, Registry.load(kernel_tree()))
    except ConfigurationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
