"""Shared helpers for the softmax kernels' registry declarations.

All five softmax kernels answer the same query -- a 2-D shape plus the axis
being normalised -- so the parameter unpacking, the axis convention and the
constants they reason about live here rather than being repeated five times.

The query parameters a softmax kernel receives are:

``shape``  the input shape (any rank; flattened to 2-D around ``dim``)
``dim``    the axis softmax runs along, following torch's convention that
           negative values count from the end

Everything a kernel actually routes on -- ``rows``, ``n``, ``width`` -- is
derived from those two by :func:`softmax_query`, so the kernels cannot disagree
about what a given ``(shape, dim)`` means.

Softmax's framework-layer adapters live here too, for the same reason the specs
live beside their kernels: the op-agnostic registry should carry no softmax
vocabulary. They register on import, and discovery imports this module, so
:func:`~ipu_apps.kernel_registry.lookup_layer` sees them.
"""

from __future__ import annotations

from dataclasses import dataclass

from ipu_apps.kernel_registry import (
    ShapeBundle,
    UnsupportedLayer,
    flatten_to_matrix,
    register_layer,
)

LANES = 128                  # datapath width every constraint is relative to
SINGLE_GROUP_MAX_ROWS = 128  # rows whose per-row scalars fit one 128-element vector

WIDE_VECTOR_ONLY = (
    "Wide-vector FP32 debug mode only (wide_vector_debug=True). These apps "
    "build on exp2/reciprocal over an FP32 vector path and have no narrow "
    "(INT8/FP8) variant."
)


@dataclass(frozen=True)
class SoftmaxQuery:
    """A softmax query reduced to what the kernels route on.

    Attributes:
        rows:    Rows of the (possibly flattened) 2-D problem.
        cols:    Columns of that problem.
        along_rows: True when softmax runs along each row (torch ``dim=1`` on a
            2-D input), False when it runs down each column (``dim=0``).
        n:       Reduction length when reducing along rows, else None.
        width:   Independent columns when reducing down columns, else None.
        bundle:  The shape bundle, carrying any flatten note.
    """

    rows: int
    cols: int
    along_rows: bool
    bundle: ShapeBundle

    @property
    def n(self) -> int | None:
        return self.cols if self.along_rows else None

    @property
    def width(self) -> int | None:
        return None if self.along_rows else self.cols

    @property
    def reduction_length(self) -> int:
        """How many elements each softmax sums over."""
        return self.cols if self.along_rows else self.rows


def softmax_query(shape, dim: int) -> SoftmaxQuery:
    """Normalise ``(shape, dim)`` into the form every softmax kernel routes on.

    Raises:
        ValueError: if ``dim`` is out of range for ``shape``, or the shape is
            rank > 2 with an interior reduction axis (which cannot be flattened
            without transposing -- see ``flatten_to_matrix``).
    """
    bundle, dim_2d, shape_2d = softmax_bundle(tuple(int(d) for d in shape), int(dim))
    rows, cols = shape_2d
    return SoftmaxQuery(
        rows=rows, cols=cols, along_rows=(dim_2d == 1), bundle=bundle
    )


def positive_dims(q: SoftmaxQuery) -> str | None:
    """Return a refusal reason if the problem has a non-positive extent."""
    if q.rows < 1:
        return f"rows ({q.rows}) must be >= 1"
    if q.cols < 1:
        return f"columns ({q.cols}) must be >= 1"
    return None


def softmax_bundle(shape, dim: int):
    """Build the shape bundle for a softmax query.

    Softmax is shape-preserving, so the output shape is derived and equal to
    the input. A rank > 2 input is flattened around ``dim`` (recorded as a note
    on the bundle, never silently).

    Returns:
        ``(bundle, dim_2d, shape_2d)``.
    """
    shape_2d, dim_2d, note = flatten_to_matrix(shape, dim)
    bundle = ShapeBundle.of(input=shape).with_shapes(
        derived={"output": shape},
        notes=(note,) if note else (),
    )
    return bundle, dim_2d, shape_2d


# -- framework-layer adapters -----------------------------------------------


@register_layer("Softmax")
def _softmax_layer(layer, input_shape):
    """``nn.Softmax(dim=...)`` -> the ``softmax`` operation.

    ``nn.Softmax`` created without an explicit ``dim`` has ``dim=None``, which
    torch itself treats as deprecated and resolves with a heuristic. Rather
    than replicate that heuristic (and risk disagreeing with the framework on
    which axis is normalised), it is refused.
    """
    if not hasattr(layer, "dim"):
        raise UnsupportedLayer(
            f"{type(layer).__name__} is missing expected attribute(s) dim; it "
            f"does not look like the layer this adapter was written for"
        )
    if layer.dim is None:
        raise UnsupportedLayer(
            "Softmax(dim=None) does not state which axis to normalise; torch "
            "resolves it with a deprecated heuristic. Construct the layer with "
            "an explicit dim."
        )
    return "softmax", {"dim": int(layer.dim), "shape": input_shape}


@register_layer("LogSoftmax", "Softmin")
def _unsupported_softmax_relatives(layer, input_shape):
    """Refuse near-neighbours of Softmax explicitly.

    These sit beside ``Softmax`` in ``torch.nn`` and share its signature, so a
    permissive adapter would route them to a softmax kernel and return
    confidently wrong numbers.
    """
    name = type(layer).__name__
    detail = {
        "LogSoftmax": "computes log(softmax(x)), not softmax(x)",
        "Softmin": "computes softmax(-x), not softmax(x)",
    }[name]
    raise UnsupportedLayer(
        f"{name} {detail}; no kernel implements it. Using a softmax kernel "
        f"here would return confidently wrong values."
    )
