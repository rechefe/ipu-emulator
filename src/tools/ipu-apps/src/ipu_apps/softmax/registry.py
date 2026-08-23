"""Softmax-flavoured front end over the generic kernel registry.

This module is a **thin translation layer**, kept so existing callers keep
working. It holds no routing rules of its own: every answer comes from
:mod:`ipu_apps.kernel_registry`, which asks the kernels themselves. The
softmax-specific vocabulary here (``axis``/``n``/``width``) is translated into
the generic ``shape``/``dim`` query and handed over.

    >>> from ipu_apps.softmax import lookup
    >>> lookup(axis="rows", n=300, rows=8).app_name
    'softmax_rows_long'

    >>> lookup(axis="columns", width=100, rows=64).app_name
    'softmax_columns'

Prefer the generic interface for new code -- it takes a framework layer, or an
op plus shapes, and works for every operation rather than just softmax::

    from ipu_apps.kernel_registry import lookup_layer, resolve

    lookup_layer(nn.Softmax(dim=1), input_shape=(32, 300))
    resolve("softmax", shape=(32, 300), dim=1)

Terminology
-----------
``axis="rows"``    softmax along each row: ``out[r, :] = softmax(x[r, :])``.
                   The reduction length is ``n`` (elements per row).
``axis="columns"`` softmax down each column: ``out[:, c] = softmax(x[:, c])``.
                   The reduction length is ``rows``; ``width`` is how many
                   independent columns ride side by side.

:data:`GAPS` lists shapes no app covers; it is currently empty -- every 1-D
softmax shape is routable at any size, subject only to the cross-cutting limit
reported as ``Verdict.caveats`` (wide-vector FP32 mode).
"""

from __future__ import annotations

from ipu_apps.kernel_registry import Verdict, boundaries, kernels, resolve
from ipu_apps.softmax._spec_support import LANES, SINGLE_GROUP_MAX_ROWS, WIDE_VECTOR_ONLY

# Kept for callers that imported them from here.
__all__ = [
    "GAPS",
    "LANES",
    "SINGLE_GROUP_MAX_ROWS",
    "Verdict",
    "catalog",
    "lookup",
    "lookup_torch",
]

# No shape is uncovered. Retained (empty) because callers assert on it.
GAPS: tuple[str, ...] = ()


def lookup(
    *,
    axis: str,
    rows: int,
    n: int | None = None,
    width: int | None = None,
) -> Verdict:
    """Return the app that computes this softmax, or why none does.

    Args:
        axis:  ``"rows"`` (softmax along each row) or ``"columns"``
               (softmax down each column).
        rows:  Number of rows in the input matrix. For ``axis="columns"``
               this is also the reduction length.
        n:     Elements per row -- the reduction length. Required for
               ``axis="rows"``.
        width: Independent columns per row. Required for ``axis="columns"``.

    Returns:
        A :class:`~ipu_apps.kernel_registry.Verdict`; truthy when an app covers
        the shape.

    Raises:
        ValueError: if ``axis`` is unknown or the axis-specific size is missing.
    """
    if axis == "rows":
        if n is None:
            raise ValueError("axis='rows' requires n (elements per row)")
        cols, dim = int(n), 1
    elif axis == "columns":
        if width is None:
            raise ValueError("axis='columns' requires width (columns per row)")
        cols, dim = int(width), 0
    else:
        raise ValueError(f"axis must be 'rows' or 'columns'; got {axis!r}")

    rows = int(rows)
    # A non-positive extent cannot be expressed as a shape, so refuse it here
    # rather than letting shape validation raise.
    if rows < 1 or cols < 1:
        bad, name = (rows, "rows") if rows < 1 else (cols, "n" if dim else "width")
        return Verdict(False, f"{name} ({bad}) must be >= 1.")

    return resolve("softmax", shape=(rows, cols), dim=dim)


def lookup_torch(shape, dim: int = -1) -> Verdict:
    """Answer a query phrased the way ``torch.softmax(x, dim)`` is called.

    Args:
        shape: The input tensor's shape. 1-D and 2-D are supported directly;
            higher ranks are flattened around ``dim`` (disclosed in the
            verdict).
        dim:   The dimension softmax is taken over, following torch's
            convention that negative values count from the end.

    Returns:
        A :class:`~ipu_apps.kernel_registry.Verdict`.

    Raises:
        ValueError: if ``dim`` is out of range for ``shape``.
    """
    dims = tuple(int(s) for s in shape)
    if not dims:
        raise ValueError("shape must have at least one dimension")
    if any(d < 1 for d in dims):
        return Verdict(False, f"every dimension of {dims} must be >= 1.")
    return resolve("softmax", shape=dims, dim=int(dim))


def catalog() -> str:
    """Render the coverage table, probed from the kernels themselves."""
    lines = [
        "softmax app coverage (all wide-vector FP32 mode)",
        "",
        "axis=rows      -- out[r,:] = softmax(x[r,:]); reduction length = n",
        "                  torch: softmax(x, dim=1) on a (rows, n) tensor",
    ]
    for b in boundaries(
        "softmax", "shape", range(1, 200), build=lambda n: (8, n), dim=1
    ):
        lines.append("  " + b.render("n"))

    lines += [
        "",
        "axis=columns   -- out[:,c] = softmax(x[:,c]); reduction length = rows",
        "                  torch: softmax(x, dim=0) on a (rows, width) tensor",
    ]
    for b in boundaries(
        "softmax", "shape", range(1, 200), build=lambda w: (8, w), dim=0
    ):
        lines.append("  " + b.render("width"))

    lines += [
        "",
        "  (row apps loop groups of <=128 rows internally: no row-count cap)",
        "",
    ]
    if GAPS:
        lines.append("Open gaps:")
        lines += [f"  - {g}" for g in GAPS]
    else:
        lines.append("Open gaps: none -- every 1-D softmax shape routes to an app.")
    lines += ["", "Cross-cutting limits:", f"  - {WIDE_VECTOR_ONLY}"]
    lines += [
        "",
        f"Kernels registered: {', '.join(k.name for k in kernels('softmax'))}",
    ]
    return "\n".join(lines)
