"""Which softmax app (if any) handles a given 1-D softmax problem.

The softmax family is five apps split across two axes -- which direction the
reduction runs, and how the reduced length relates to the 128-lane datapath.
Picking the right one by hand means re-reading five docstrings and their
constructor guards. :func:`lookup` does it for you:

    >>> from ipu_apps.softmax import lookup
    >>> lookup(axis="rows", n=300, rows=8).app_name
    'softmax_rows_long'

    >>> lookup(axis="columns", width=100, rows=64).app_name
    'softmax_columns'

Every answer is derived from the SAME constraint table the apps enforce in
their own ``__init__``, so a :attr:`Verdict.supported` answer will construct;
an unsupported one says why instead of failing later.

Terminology
-----------
``axis="rows"``    softmax along each row: ``out[r, :] = softmax(x[r, :])``.
                   The reduction length is ``n`` (elements per row).
``axis="columns"`` softmax down each column: ``out[:, c] = softmax(x[:, c])``.
                   The reduction length is ``rows``; ``width`` is how many
                   independent columns ride side by side.

:data:`GAPS` lists shapes no app covers; it is currently empty -- every 1-D
softmax shape is routable at any size, subject only to the cross-cutting limit
reported as :attr:`Verdict.caveats` (wide-vector FP32 mode).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

LANES = 128  # the datapath width every constraint below is relative to

# Per-row scalars (maxvec/rvec) hold one slot per row in a single 128-lane
# vector, so every row app processes at most this many rows per group and then
# re-runs all four passes on the next group. Internal to the kernels -- it caps
# group size, not the total row count.
SINGLE_GROUP_MAX_ROWS = 128


@dataclass(frozen=True)
class Verdict:
    """The answer to one :func:`lookup` query.

    Attributes:
        supported:   True if an app covers this shape.
        app_name:    Package name under ``ipu_apps.softmax`` (None if not covered).
        app_class:   Importable class path (None if not covered).
        kwargs:      Constructor keyword arguments to pass, beyond the common
                     ``inst_path`` / ``input_path`` / ``output_path``.
        reason:      Why this app was chosen, or why nothing covers the shape.
        caveats:     Non-blocking limits that still apply if you proceed.
    """

    supported: bool
    reason: str
    app_name: str | None = None
    app_class: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    caveats: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.supported

    def describe(self) -> str:
        """Human-readable multi-line summary (what ``__main__`` prints)."""
        head = "SUPPORTED" if self.supported else "NOT SUPPORTED"
        lines = [f"{head}: {self.reason}"]
        if self.app_name:
            args = ", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())
            lines.append(f"  app:  {self.app_name}")
            lines.append(f"  use:  {self.app_class}(inst_path=..., input_path=..., "
                         f"output_path=..., {args})")
        for c in self.caveats:
            lines.append(f"  note: {c}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Caveats that apply across apps (real limits, verified against the code)
# --------------------------------------------------------------------------

_WIDE_VECTOR_ONLY = (
    "Wide-vector FP32 debug mode only (wide_vector_debug=True). These apps "
    "build on exp2/reciprocal over an FP32 vector path and have no narrow "
    "(INT8/FP8) variant."
)


def _idle_lane_caveat(width: int, padded: int) -> str:
    return (
        f"width {width} pads to {padded} lanes, so {padded - width} of every "
        f"{padded} lanes sit idle ({width / padded:.0%} utilisation). A chunk "
        f"costs the same regardless of how many lanes carry real columns, so "
        f"this runs at the cost of width {padded}. Packing cannot help below "
        f"128 either -- the next power of two is 128, i.e. one row per vector."
    )


# --------------------------------------------------------------------------
# Known coverage gaps -- shapes no app currently handles
# --------------------------------------------------------------------------

GAPS: tuple[str, ...] = ()


def _rows_axis(n: int, rows: int) -> Verdict:
    if n < 1:
        return Verdict(False, f"n ({n}) must be >= 1.")
    if rows < 1:
        return Verdict(False, f"rows ({rows}) must be >= 1.")

    if n == LANES:
        # The only row app with a >128-row group loop.
        return Verdict(
            True,
            f"n == {LANES} exactly: the full-width row kernel, which also "
            f"loops groups of {LANES} rows, so any row count works.",
            app_name="softmax_rows",
            app_class="ipu_apps.softmax.softmax_rows.SoftmaxRowsApp",
            kwargs={"rows": rows},
            caveats=(_WIDE_VECTOR_ONLY,),
        )

    if n < LANES:
        ps = 16
        while ps < n:
            ps *= 2
        p = LANES // ps
        return Verdict(
            True,
            f"n ({n}) < {LANES}: packed row kernel, partition size ps={ps}, "
            f"P={p} logical rows per 128-lane chunk, in groups of "
            f"{SINGLE_GROUP_MAX_ROWS // p} chunks.",
            app_name="softmax_rows_partial",
            app_class="ipu_apps.softmax.softmax_rows_partial.SoftmaxRowsPartialApp",
            kwargs={"n": n, "rows": rows},
            caveats=(_WIDE_VECTOR_ONLY,),
        )

    # n > LANES
    full, tail = divmod(n, LANES)
    if tail:
        shape = (f"{full} full chunks + {tail}-element tail, reduced with a "
                 f"running cross-chunk AGG")
    else:
        shape = (f"exactly {full} full chunks (no tail chunk), reduced with a "
                 f"running cross-chunk AGG")
    return Verdict(
        True,
        f"n ({n}) > {LANES}: {shape}.",
        app_name="softmax_rows_long",
        app_class="ipu_apps.softmax.softmax_rows_long.SoftmaxRowsLongApp",
        kwargs={"n": n, "rows": rows},
        caveats=(_WIDE_VECTOR_ONLY,),
    )


def _columns_axis(width: int, rows: int) -> Verdict:
    if width < 1:
        return Verdict(False, f"width ({width}) must be >= 1.")
    if rows < 1:
        return Verdict(False, f"rows ({rows}) must be >= 1.")

    if width <= 64:
        w_pad = 16
        while w_pad < width:
            w_pad *= 2
        rpv = LANES // w_pad
        return Verdict(
            True,
            f"width ({width}) <= 64: packed column kernel, {rpv} rows packed "
            f"per 128-lane vector (padded width {w_pad}).",
            app_name="softmax_columns_packed",
            app_class="ipu_apps.softmax.softmax_columns_packed.SoftmaxColumnsPackedApp",
            kwargs={"rows": rows, "width": width},
            caveats=(_WIDE_VECTOR_ONLY,),
        )

    cpr = (width + LANES - 1) // LANES
    padded = cpr * LANES
    caveats = [_WIDE_VECTOR_ONLY]
    if padded != width:
        caveats.append(_idle_lane_caveat(width, padded))

    return Verdict(
        True,
        f"width ({width}) >= 65: per-lane running ACC reduce down the rows "
        f"over {cpr} chunk-column(s), no AGG and no row-group cap.",
        app_name="softmax_columns",
        app_class="ipu_apps.softmax.softmax_columns.SoftmaxColumnsApp",
        kwargs={"rows": rows, "width": width},
        caveats=tuple(caveats),
    )


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
        A :class:`Verdict`; truthy when an app covers the shape.

    Raises:
        ValueError: if ``axis`` is unknown or the axis-specific size is missing.
    """
    if axis == "rows":
        if n is None:
            raise ValueError("axis='rows' requires n (elements per row)")
        return _rows_axis(int(n), int(rows))
    if axis == "columns":
        if width is None:
            raise ValueError("axis='columns' requires width (columns per row)")
        return _columns_axis(int(width), int(rows))
    raise ValueError(f"axis must be 'rows' or 'columns'; got {axis!r}")


def lookup_torch(shape, dim: int = -1) -> Verdict:
    """Answer a query phrased the way ``torch.softmax(x, dim)`` is called.

    This is :func:`lookup` with PyTorch's calling convention: you pass the
    tensor's ``shape`` and the ``dim`` being normalised, exactly as you would
    write the torch call, instead of translating to the row/column model
    yourself::

        >>> import torch                                   # doctest: +SKIP
        >>> y = torch.softmax(torch.randn(32, 300), dim=1)  # doctest: +SKIP
        >>> lookup_torch((32, 300), dim=1).app_name
        'softmax_rows_long'

    Which is why the shape is required and not just the size of ``dim``: the
    apps reduce one axis while the OTHER axis rides along as independent
    problems, and the cost, the layout, and sometimes the app itself depend on
    how many of those there are. ``dim`` alone cannot pick a kernel.

    Args:
        shape: The input tensor's shape, as a sequence of ints. 1-D and 2-D are
            supported. A 1-D shape ``(n,)`` is treated as ``(1, n)``.
        dim:   The dimension softmax is taken over, following torch's
            convention that negative values count from the end. Defaults to
            ``-1`` (the last dim).

    Returns:
        A :class:`Verdict`, truthy when an app covers the shape.

    Raises:
        ValueError: if the shape isn't 1-D or 2-D, or ``dim`` is out of range
            for it.
    """
    dims = tuple(int(s) for s in shape)

    if len(dims) == 1:
        # A vector has exactly one axis, so dim 0 (== -1) is the only legal
        # choice and it always means "along the vector". Validate against the
        # 1-D rank, then treat it as a single row -- promoting to (1, n) FIRST
        # would silently reinterpret dim=0 as the column axis of that matrix.
        if dim not in (0, -1):
            raise ValueError(
                f"dim {dim} is out of range for a 1-D shape {tuple(shape)!r}"
            )
        return _rows_axis(dims[0], 1)

    if len(dims) != 2:
        raise ValueError(
            f"shape {tuple(shape)!r} has {len(dims)} dimensions; these apps "
            f"cover 1-D and 2-D inputs only. For a higher-rank tensor, flatten "
            f"the non-softmax dimensions into one axis first -- softmax over "
            f"`dim` is independent across every other dimension."
        )

    if not -2 <= dim < 2:
        raise ValueError(
            f"dim {dim} is out of range for a 2-D shape {tuple(shape)!r}"
        )
    dim %= 2                         # torch semantics: -1 -> last

    rows, cols = dims
    if dim == 1:
        # Reducing along each row: n = row length, `rows` rows ride along.
        return _rows_axis(cols, rows)
    # Reducing down each column: reduction length = rows, `cols` columns ride along.
    return _columns_axis(cols, rows)


def catalog() -> str:
    """Render the whole coverage table -- what exists, and what doesn't."""
    lines = [
        "softmax app coverage (all wide-vector FP32 mode)",
        "",
        "axis=rows      -- out[r,:] = softmax(x[r,:]); reduction length = n",
        "                  torch: softmax(x, dim=1) on a (rows, n) tensor",
        "  n < 128                    softmax_rows_partial   (packed, P=128/ps rows/chunk)",
        "  n == 128                   softmax_rows           (full-width rows)",
        "  n > 128                    softmax_rows_long      (running cross-chunk reduce;",
        "                                                     no tail chunk when n % 128 == 0)",
        "  (all three loop groups of <=128 rows internally: no row-count cap)",
        "",
        "axis=columns   -- out[:,c] = softmax(x[:,c]); reduction length = rows",
        "                  torch: softmax(x, dim=0) on a (rows, width) tensor",
        "  width <= 64                softmax_columns_packed (rpv=128/W rows per vector)",
        "  width >= 65                softmax_columns        (no row-count cap; widths not a",
        "                                                     multiple of 128 leave lanes idle)",
        "",
    ]
    if GAPS:
        lines.append("Open gaps:")
        lines += [f"  - {g}" for g in GAPS]
    else:
        lines.append("Open gaps: none -- every 1-D softmax shape routes to an app.")
    lines += ["", "Cross-cutting limits:", f"  - {_WIDE_VECTOR_ONLY}"]
    return "\n".join(lines)
