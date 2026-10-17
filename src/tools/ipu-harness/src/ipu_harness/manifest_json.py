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
from ipu_harness.family import REGISTRY_NAME
from ipu_harness.manifest import TOML_NAME
from ipu_harness.registry import Registry, kernel_tree


def _skipped(declared: Mapping[str, Any], invalid: Mapping[Path, ConfigurationError]) -> set[str]:
    """The declared kernels that failed to load: a ``kernel.toml``'s folder, an
    entry, or every entry of a ``registry.toml`` that did not read."""
    skipped = set()
    for path in invalid:
        if path.name == REGISTRY_NAME:
            family = path.parent.name
            skipped |= {
                n for n, t in declared.items() if t.get("registry") and t["family"] == family
            }
        elif path.name == TOML_NAME:
            skipped.add(path.parent.name)
        else:
            skipped.add(path.name)
    return skipped


def build(targets: Mapping[str, Any], registry: Registry) -> dict[str, Any]:
    """The manifest, from the targets file and the registry.

    Raises:
        ConfigurationError: The targets and the registry name different kernels.
    """
    declared = set(targets["kernels"])
    skipped = _skipped(targets["kernels"], registry.invalid)
    every = {**registry.templates, **registry.manifests}
    loaded = set(every)
    if declared - skipped != loaded:
        raise ConfigurationError(
            "kernels/BUILD.bazel and the kernel tree disagree -- targets without a kernel: "
            f"{sorted(declared - skipped - loaded) or 'none'}; kernels without a target: "
            f"{sorted(loaded - declared) or 'none'}"
        )
    kernels: dict[str, Any] = {}
    operations: dict[str, set[str]] = {}
    for name, manifest in sorted(every.items()):
        bazel = targets["kernels"][name]
        if name in registry.manifests:
            operations.setdefault(manifest.op, set()).update(manifest.params)
        kernels[name] = {
            "op": manifest.op,
            "variant": name,
            "tags": [],
            "family": bazel["family"],
            # E06-17: routing skips a template (its entries are routed);
            # an entry names its template and may carry a one-line doc.
            "routed": name in registry.manifests,
            "template": bazel.get("template"),
            "doc": manifest.doc,
            "asm": f"{bazel['dir']}/{manifest.asm.name}",
            "targets": {k: bazel[k] for k in ("run", "test", "nightly")},
            "formats": [f.value for f in manifest.formats],
            # "options" are the command-line flags a case adds (the legacy
            # manifest's descriptors): a harness case adds none, its values are
            # fixed by its name. "values" are those values.
            "cases": {
                case.name: {
                    "options": {},
                    "max_cycles": None,
                    "values": dict(case.values),
                    "tier": case.tier,
                }
                for case in expand(manifest)
            },
        }
    return {
        "kernels": kernels,
        "families": dict(targets["families"]),
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
