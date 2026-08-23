"""The registry itself: match a query against every declared kernel.

Resolution is deliberately dumb, and that is the point. The registry holds no
per-operation knowledge -- no ``if n < 128`` lives here. It asks every kernel
registered for the operation whether it handles the query, and picks the
cheapest one that says yes. All domain knowledge stays in the kernels.

When several kernels claim the same query the winner is decided by each
kernel's declared ``cost``, never by discovery order. Order-dependent
resolution is the classic way a plugin registry becomes quietly
non-deterministic: the answer changes because a file was renamed.
"""

from __future__ import annotations

import threading
from typing import Any

from ipu_apps.kernel_registry.discovery import Discovered, discover
from ipu_apps.kernel_registry.shapes import MalformedQuery
from ipu_apps.kernel_registry.spec import KernelSpec, Verdict

_LOCK = threading.Lock()
_CACHE: dict[str, Discovered] = {}


def load(package: str = "ipu_apps", *, refresh: bool = False) -> Discovered:
    """Discover kernels once per package and memoise the result.

    Discovery imports the whole app tree, which is far too slow to repeat on
    every lookup.
    """
    with _LOCK:
        if refresh or package not in _CACHE:
            _CACHE[package] = discover(package)
        return _CACHE[package]


def kernels(op: str | None = None, *, package: str = "ipu_apps") -> tuple[KernelSpec, ...]:
    """Every declared kernel, optionally filtered to one operation."""
    found = load(package).specs
    if op is None:
        return tuple(sorted(found, key=lambda s: (s.op, s.name)))
    return tuple(sorted((s for s in found if s.op == op), key=lambda s: s.name))


def operations(*, package: str = "ipu_apps") -> tuple[str, ...]:
    """Every operation that has at least one kernel."""
    return tuple(sorted({s.op for s in load(package).specs}))


def resolve(op: str, *, package: str = "ipu_apps", **params: Any) -> Verdict:
    """Find the best kernel for ``op`` with ``params``.

    Args:
        op:      Operation name, e.g. ``"softmax"``.
        package: Root package to search.
        **params: Operation-specific parameters. Passed verbatim to each
            candidate kernel's ``supports``/``build``/``explain``.

    Returns:
        A :class:`Verdict`, truthy when a kernel covers the query. When nothing
        covers it, the reason aggregates every candidate's refusal, so the
        caller learns what was wrong rather than just that it failed.
    """
    candidates = kernels(op, package=package)
    if not candidates:
        known = ", ".join(operations(package=package)) or "none"
        return Verdict(
            False,
            f"no kernels are registered for operation {op!r} (known operations: {known})",
        )

    # Note there is deliberately no ``except KeyError`` here. Specs index
    # ``**params``, so an omitted parameter would surface that way -- but it is
    # indistinguishable from a KeyError raised *inside* a callback, which is a
    # bug in that kernel and must not be quietly downgraded to "unsupported".
    # ``KernelSpec.requires`` states the contract instead, and ``check`` turns a
    # missing parameter into a refusal before any callback runs.
    accepted: list[tuple[float, KernelSpec, str]] = []
    refusals: list[str] = []
    for spec in candidates:
        try:
            support = spec.check(**params)
        except TypeError as exc:
            # The kernel does not understand these parameters at all -- a
            # missing/extra keyword. That is a refusal, not a crash, but it is
            # worth reporting precisely because it usually means the caller
            # named a parameter wrongly.
            refusals.append(f"{spec.name}: does not accept these parameters ({exc})")
            continue
        except MalformedQuery:
            # The question itself is invalid (a dim outside the input's rank),
            # so no kernel could ever answer it. Propagate rather than
            # reporting it as missing coverage -- a typo is not a gap.
            raise
        except ValueError as exc:
            # A shape this kernel cannot interpret, but a well-formed question.
            # Asking "does anything support this?" should answer no with the
            # reason rather than raise: callers branch on the verdict.
            refusals.append(f"{spec.name}: {exc}")
            continue
        if support.ok:
            accepted.append((float(spec.cost(**params)), spec, support.reason))
        else:
            refusals.append(f"{spec.name}: {support.reason}")

    if not accepted:
        detail = (
            "; ".join(_distinct(refusals, len(candidates)))
            if refusals else "no kernel claimed it"
        )
        return Verdict(False, f"no {op} kernel covers this query -- {detail}")

    accepted.sort(key=lambda item: (item[0], item[1].name))
    _, best, support_reason = accepted[0]
    others = tuple(spec.name for _, spec, _ in accepted[1:])

    # A kernel that said yes *with* a reason has said something specific about
    # this query; only fall back to the generic explanation when it did not.
    reason = support_reason or best.explain(**params)

    # A kernel may normalise the query's shapes (flattening a rank>2 input,
    # deriving an output shape). Prefer what it reports, so any reinterpretation
    # reaches the caller instead of being lost between the spec and the verdict.
    caveats = list(best.caveats(**params))
    shapes = params.get("shapes")
    if best.bundle is not None:
        try:
            shapes = best.bundle(**params) or shapes
        except Exception as exc:
            # A broken bundle helper must not break resolution -- but it must
            # not silently cost the caller a disclosure either. Everything else
            # here refuses to hide a reinterpretation; swallowing this one
            # would let a flatten note vanish without trace.
            caveats.append(
                f"could not report the shapes as {best.name} understands them "
                f"({type(exc).__name__}: {exc}); any reinterpretation this "
                f"kernel applies is NOT disclosed below"
            )
    return Verdict(
        supported=True,
        reason=reason,
        kernel=best,
        kwargs=best.build(**params),
        shapes=shapes,
        caveats=tuple(caveats),
        alternatives=others,
    )


def _distinct(reasons: list[str], candidates: int) -> list[str]:
    """Collapse refusals that differ only by which kernel voiced them.

    When a query fails to normalise at all -- a softmax around an interior axis,
    say -- every candidate refuses with the same sentence, and repeating it once
    per kernel buries the one thing the caller needs to read.
    """
    order: list[str] = []
    voices: dict[str, list[str]] = {}
    for entry in reasons:
        name, _, detail = entry.partition(": ")
        if detail not in voices:
            order.append(detail)
            voices[detail] = []
        voices[detail].append(name)

    collapsed = []
    for detail in order:
        names = voices[detail]
        if len(names) == 1:
            collapsed.append(f"{names[0]}: {detail}")
        elif len(names) == candidates:
            collapsed.append(f"every kernel ({', '.join(names)}): {detail}")
        else:
            collapsed.append(f"{', '.join(names)}: {detail}")
    return collapsed
