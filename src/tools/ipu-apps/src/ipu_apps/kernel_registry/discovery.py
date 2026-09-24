"""Finding every kernel that has declared itself, at any nesting depth.

Kernels are not listed anywhere central. Every kernel has its own directory
and keeps its harness class and ``SPEC`` together in a file named ``app.py``,
with an empty ``__init__.py`` alongside it -- discovery
walks the ``ipu_apps`` package tree and imports exactly the modules named
``app``, nothing else. Adding a kernel is then purely additive: no
registration list to edit, no import to remember.

Two properties this has to hold:

* **Arbitrary depth.** Family kernels sit two levels down
  (``kernels/softmax/softmax_rows/``, ``kernels/convolutions/conv1x1/``), and
  nothing fixes that depth. Recursion is not optional.
* **Tolerance of broken or half-present packages.** A working tree can easily
  contain directories that are not importable -- stale ``__pycache__`` shells
  left behind by a branch switch, a kernel mid-authoring, an optional
  dependency that is not installed. Discovery records those as
  :class:`SkippedModule` and carries on. A registry that raises on import of an
  unrelated half-finished app would be worse than useless in exactly the
  situation people need it.

Targeting ``app.py`` specifically (rather than importing every submodule and
checking each for a ``SPEC``) means a shared helper module -- a family's ``cases.py``,
``prepare`` code, whatever a kernel family needs -- is never imported by
discovery just to find it has no spec. There is no name-based skip-list to
keep in sync with the codebase's naming conventions; only the one convention
every kernel already follows matters.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
from dataclasses import dataclass
from types import ModuleType

from ipu_apps.kernel_registry.spec import KernelSpec


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


def _specs_in(module: ModuleType) -> list[KernelSpec]:
    spec = getattr(module, "SPEC", None)
    return [spec] if isinstance(spec, KernelSpec) else []


def discover(package: str = "ipu_apps") -> Discovered:
    """Import ``package`` recursively and collect every declared kernel spec.

    Args:
        package: Root package to walk. Defaults to the whole app tree; pass a
            subpackage (``"ipu_apps.kernels.softmax"``) to scope discovery.

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

    def unimportable_package(name: str) -> None:
        # walk_packages imports each package to recurse into it, and calls this
        # from inside its except block. Without it a broken family __init__
        # would silently hide every kernel beneath it.
        exc = sys.exc_info()[1]
        skipped.append(SkippedModule(name, f"{type(exc).__name__}: {exc}"))

    for info in pkgutil.walk_packages(paths, prefix=f"{package}.", onerror=unimportable_package):
        if info.name.rsplit(".", 1)[-1] != "app":
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
