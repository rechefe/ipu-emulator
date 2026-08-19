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
"""

from __future__ import annotations

from dataclasses import dataclass

from ipu_apps.kernel_registry import ShapeBundle, softmax_bundle

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
