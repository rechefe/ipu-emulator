"""``when`` conditions: a tensor or binding that exists only in some cases (E06 gap G6).

A condition names a ``bool`` parameter, or ``not`` one: ``when = "bias"``,
``when = "not bias"``. :func:`narrowed` gives a case's view of a kernel: only
the tensors and ``[abi]`` bindings whose condition holds. ``plan_case`` hands
that view to every later stage, so placement, packing, inputs, goldens,
registers and the runner see only what exists.
"""

from __future__ import annotations

import weakref
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ipu_harness.manifest import KernelManifest

__all__ = ["Condition", "narrowed"]

_NOT = "not "


@dataclass(frozen=True)
class Condition:
    """``param`` holds (or, ``negated``, does not).

    Attributes:
        param: A ``bool`` parameter's name.
        negated: Whether the condition is ``not param``.
    """

    param: str
    negated: bool = False

    @classmethod
    def parse(cls, text: str) -> Condition:
        """``"bias"`` or ``"not bias"`` (any spacing)."""
        words = " ".join(text.split())
        name = words.removeprefix(_NOT)
        return cls(name, name != words)

    def holds(self, params: Mapping[str, object]) -> bool:
        """Whether it holds for a case's values (defaults filled)."""
        return bool(params[self.param]) is not self.negated

    def excludes(self, other: Condition | None) -> bool:
        """Whether it and ``other`` never hold together."""
        return other is not None and other.param == self.param and other.negated != self.negated

    def __str__(self) -> str:
        return _NOT + self.param if self.negated else self.param


def _holds(condition: Condition | None, params: Mapping[str, object]) -> bool:
    return condition is None or condition.holds(params)


#: A manifest's narrowed views, per (absent tensors, inactive bindings); held
#: weakly, so a manifest no one keeps is freed with its views.
_VIEWS: weakref.WeakKeyDictionary[
    KernelManifest, dict[tuple[frozenset[str], frozenset[str]], KernelManifest]
] = weakref.WeakKeyDictionary()


def narrowed(manifest: KernelManifest, params: Mapping[str, object]) -> KernelManifest:
    """``manifest`` as one case sees it: only the tensors and bindings whose
    condition holds for ``params``.

    Args:
        manifest: The kernel.
        params: The case's checked values, defaults filled.

    Returns:
        ``manifest`` itself when every condition holds (or it has none), else
        one shared view per answer.
    """
    absent = frozenset(n for n, t in manifest.tensors.items() if not _holds(t.when, params))
    inactive = frozenset(n for n, c in manifest.abi_when.items() if not _holds(c, params))
    if not absent and not inactive:
        return manifest
    views = _VIEWS.setdefault(manifest, {})
    key = (absent, inactive)
    if key not in views:
        abi = {n: reg for n, reg in manifest.abi.items() if n not in inactive}
        views[key] = replace(
            manifest,
            tensors=MappingProxyType(
                {n: t for n, t in manifest.tensors.items() if n not in absent}
            ),
            abi=MappingProxyType(abi),
            abi_offsets=MappingProxyType(
                {n: o for n, o in manifest.abi_offsets.items() if n in abi}
            ),
            abi_when=MappingProxyType({n: c for n, c in manifest.abi_when.items() if n in abi}),
        )
    return views[key]
