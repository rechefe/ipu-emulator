"""Shape bundles: the tensors an operation reads and writes, keyed by role.

A kernel query is never "one shape". Even the simplest op has an input and an
output; a matmul has two independent inputs; a convolution has an input, a
weight, usually a bias, and an output whose extent follows from all three.
:class:`ShapeBundle` is the single container all of that travels in, so the
registry's query envelope does not have to change shape per operation.

Roles are plain strings rather than an enum: ``"input"``/``"output"`` are
conventional, ``"weight"``/``"bias"`` appear for parameterised layers, and an
op is free to use its own (``matmul`` uses ``"a"``/``"b"``). New operations add
roles without touching this module.

Provenance matters as much as the shapes. A shape the caller *asserted* and a
shape the registry *derived* from a layer's parameters are different kinds of
claim: if a derivation is wrong, the resulting verdict must not read as though
the caller asked for it. :meth:`ShapeBundle.derived_roles` records which is
which, and verdicts surface it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

Shape = tuple[int, ...]


class MalformedQuery(ValueError):
    """The query itself is invalid, independent of which kernels exist.

    Distinct from "no kernel handles this shape": a ``dim`` outside the input's
    rank is a mistake in the *question*, and no amount of adding kernels would
    make it answerable. Those propagate to the caller instead of being folded
    into a verdict, so a typo is not reported as missing coverage.
    """


INPUT = "input"
OUTPUT = "output"
WEIGHT = "weight"
BIAS = "bias"


def _normalise(shape: Iterable[int]) -> Shape:
    dims = tuple(int(d) for d in shape)
    if not dims:
        raise ValueError("a shape must have at least one dimension")
    if any(d < 0 for d in dims):
        raise ValueError(f"shape {dims} has a negative dimension")
    return dims


@dataclass(frozen=True)
class ShapeBundle:
    """The shapes of every tensor an operation touches, keyed by role.

    Attributes:
        shapes:        role -> shape, e.g. ``{"input": (32, 300)}``.
        derived_roles: roles this registry computed rather than received. Used
                       so a verdict can distinguish "you told me" from "I
                       worked it out".
        notes:         human-readable record of any reinterpretation applied on
                       the caller's behalf -- notably flattening a >2-D input.
                       Surfaced in the verdict so a silent reshape is
                       impossible.
    """

    shapes: Mapping[str, Shape]
    derived_roles: frozenset[str] = field(default_factory=frozenset)
    notes: tuple[str, ...] = ()

    @classmethod
    def of(cls, **shapes: Iterable[int]) -> "ShapeBundle":
        """Build a bundle from keyword roles: ``ShapeBundle.of(input=(32, 300))``."""
        return cls({role: _normalise(s) for role, s in shapes.items()})

    def __getitem__(self, role: str) -> Shape:
        return self.shapes[role]

    def __contains__(self, role: str) -> bool:
        return role in self.shapes

    def get(self, role: str, default: Shape | None = None) -> Shape | None:
        return self.shapes.get(role, default)

    def with_shapes(
        self,
        *,
        derived: Mapping[str, Iterable[int]] | None = None,
        asserted: Mapping[str, Iterable[int]] | None = None,
        notes: Iterable[str] = (),
    ) -> "ShapeBundle":
        """Return a copy with extra roles added.

        ``derived`` shapes are recorded in :attr:`derived_roles`; ``asserted``
        ones are treated as caller-supplied.
        """
        merged = dict(self.shapes)
        new_derived = set(self.derived_roles)
        for role, s in (derived or {}).items():
            merged[role] = _normalise(s)
            new_derived.add(role)
        for role, s in (asserted or {}).items():
            merged[role] = _normalise(s)
            new_derived.discard(role)
        return ShapeBundle(merged, frozenset(new_derived), self.notes + tuple(notes))

    def describe(self) -> str:
        """One-line rendering, marking derived roles so provenance is visible."""
        parts = []
        for role in sorted(self.shapes):
            mark = "*" if role in self.derived_roles else ""
            parts.append(f"{role}{mark}={tuple(self.shapes[role])}")
        return ", ".join(parts)


def flatten_to_matrix(shape: Shape, dim: int) -> tuple[Shape, int, str | None]:
    """Collapse a >2-D shape to 2-D around the axis ``dim`` operates on.

    An elementwise-along-one-axis op (softmax being the motivating case) is
    independent across every other axis, so a rank-N input is exactly a batch
    of 2-D problems. Rather than refusing those, the registry flattens and
    *says so* -- the returned note is carried into the verdict, because a
    reshape the caller did not ask for must never be silent.

    Returns:
        ``(shape_2d, dim_2d, note)`` where ``note`` is None when the input was
        already 2-D (or 1-D, which is promoted to a single row).
    """
    ndim = len(shape)
    if not -ndim <= dim < ndim:
        raise MalformedQuery(
            f"dim {dim} is out of range for a rank-{ndim} shape {tuple(shape)}"
        )
    dim %= ndim

    if ndim == 1:
        return (1, shape[0]), 1, None
    if ndim == 2:
        return shape, dim, None

    reduced = shape[dim]
    leading = 1
    for i, d in enumerate(shape):
        if i != dim:
            leading *= d

    # Only a trailing (or leading) reduced axis flattens without transposing;
    # a middle axis would need the other dims re-ordered around it first, and
    # silently transposing a caller's data is exactly what this must not do.
    if dim == ndim - 1:
        note = (
            f"flattened {tuple(shape)} -> {(leading, reduced)}: the operation "
            f"runs along dim {dim} (the last axis) and is independent across "
            f"every other axis, so a rank-{ndim} input is a batch of 2-D "
            f"problems"
        )
        return (leading, reduced), 1, note
    if dim == 0:
        note = (
            f"flattened {tuple(shape)} -> {(reduced, leading)}: the operation "
            f"runs down dim 0 and is independent across every other axis, so a "
            f"rank-{ndim} input is a batch of 2-D problems"
        )
        return (reduced, leading), 0, note

    raise ValueError(
        f"dim {dim} is an interior axis of a rank-{ndim} shape {tuple(shape)}; "
        f"flattening around it would require transposing the other axes, which "
        f"silently reinterprets the caller's memory layout. Move the softmax "
        f"axis to the first or last dimension first (e.g. with permute/movedim)."
    )
