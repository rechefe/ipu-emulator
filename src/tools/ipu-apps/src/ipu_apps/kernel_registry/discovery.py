"""Finding every kernel that has declared itself, at any nesting depth.

Kernels are not listed anywhere central. Discovery walks the ``ipu_apps``
package tree and collects every module-level ``SPEC`` (or ``SPECS`` for a
module declaring several). Adding a kernel is then purely additive: no
registration list to edit, no import to remember.

Two properties this has to hold:

* **Arbitrary depth.** ``softmax`` nests kernels one level down, but the
  convolution family nests them three (``convolutions_universal/conv/
  conv_universal/``). Recursion is not optional.
* **Tolerance of broken or half-present packages.** A working tree can easily
  contain directories that are not importable -- stale ``__pycache__`` shells
  left behind by a branch switch, a kernel mid-authoring, an optional
  dependency that is not installed. Discovery records those as
  :class:`SkippedModule` and carries on. A registry that raises on import of an
  unrelated half-finished app would be worse than useless in exactly the
  situation people need it.
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from types import ModuleType

from ipu_apps.kernel_registry.spec import KernelSpec

# Subpackage names that never contain kernel declarations. Skipping them keeps
# discovery from importing test/benchmark code (which is slow, and may pull in
# optional dependencies) just to find specs.
_SKIP_PARTS = frozenset({"benchmark", "test", "tests", "__pycache__"})


@dataclass(frozen=True)
class SkippedModule:
    """A module discovery could not import, and why.

    Kept rather than discarded: silent skipping would let a kernel vanish from
    coverage because of an unrelated typo, and nobody would notice.
    """

    module: str
    error: str


@dataclass(frozen=True)
class Discovered:
    specs: tuple[KernelSpec, ...]
    skipped: tuple[SkippedModule, ...]


def _is_skippable(module_name: str) -> bool:
    return any(part in _SKIP_PARTS or part.startswith("test_") for part in module_name.split("."))


def _specs_in(module: ModuleType) -> list[KernelSpec]:
    found: list[KernelSpec] = []
    single = getattr(module, "SPEC", None)
    if isinstance(single, KernelSpec):
        found.append(single)
    for extra in getattr(module, "SPECS", ()) or ():
        if isinstance(extra, KernelSpec):
            found.append(extra)
    return found


def discover(package: str = "ipu_apps") -> Discovered:
    """Import ``package`` recursively and collect every declared kernel spec.

    Args:
        package: Root package to walk. Defaults to the whole app tree; pass a
            subpackage (``"ipu_apps.softmax"``) to scope discovery.

    Returns:
        A :class:`Discovered` holding the specs found and the modules skipped.
    """
    specs: dict[str, KernelSpec] = {}
    skipped: list[SkippedModule] = []

    try:
        root = importlib.import_module(package)
    except Exception as exc:  # pragma: no cover - only if the root is broken
        return Discovered((), (SkippedModule(package, f"{type(exc).__name__}: {exc}"),))

    for spec in _specs_in(root):
        specs[spec.name] = spec

    paths = getattr(root, "__path__", None)
    if paths is None:
        return Discovered(tuple(specs.values()), ())

    for info in pkgutil.walk_packages(paths, prefix=f"{package}."):
        if _is_skippable(info.name):
            continue
        try:
            module = importlib.import_module(info.name)
        except Exception as exc:
            # An unimportable module must not take the whole registry down;
            # record it so it can still be reported as a coverage hole.
            skipped.append(SkippedModule(info.name, f"{type(exc).__name__}: {exc}"))
            continue
        for spec in _specs_in(module):
            if spec.name in specs and specs[spec.name] is not spec:
                raise ValueError(
                    f"duplicate kernel name {spec.name!r}: declared both in "
                    f"{specs[spec.name].app_class.__module__} and "
                    f"{spec.app_class.__module__}. Kernel names must be unique "
                    f"-- they identify the kernel in verdicts and coverage."
                )
            specs[spec.name] = spec

    return Discovered(tuple(specs.values()), tuple(skipped))
