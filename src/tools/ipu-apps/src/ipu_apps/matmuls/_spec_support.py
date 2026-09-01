"""Shared helpers for the matmul kernels' registry declarations.

All sixteen matmul kernels answer the same query -- two 2-D operand shapes --
so the parameter unpacking and the vocabulary they reason about (``m``, ``k``,
``n``) live here rather than being repeated sixteen times.

Every kernel in :mod:`ipu_apps.matmuls` computes ``C = A @ W^T``:

``shape_a``    the input matrix's shape, ``(M, K)``
``shape_b_t``  the weight matrix's shape *as stored*, ``(N, K)`` -- W is kept
               output-major (row ``n`` holds the K inputs feeding output n),
               the same convention the fully-connected layer uses, so no
               transpose crosses the registry boundary

``M`` is the row count of A (and C); ``K`` is the shared contraction length;
``N`` is the row count of W (and the column count of C). Unlike softmax's
kernels, which cover a *range* of row counts, every matmul kernel here is a
single fixed (M, K, N) triple with no padding or chunking tolerance -- the
match is exact or refused.
"""

from __future__ import annotations

from dataclasses import dataclass

from ipu_apps.kernel_registry import ShapeBundle

OP = "matmul"


@dataclass(frozen=True)
class MatmulQuery:
    """A matmul query reduced to what the kernels route on.

    Attributes:
        m:      Rows of A (and of the output C).
        k:      Shared contraction length (cols of A, cols of W).
        n:      Rows of W (and columns of the output C).
        bundle: The shape bundle, carrying the input/weight shapes and the
                derived output shape.
    """

    m: int
    k: int
    n: int
    bundle: ShapeBundle

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.m, self.k, self.n)


def matmul_query(shape_a, shape_b_t) -> MatmulQuery:
    """Normalise ``(shape_a, shape_b_t)`` into what every matmul kernel routes on.

    ``shape_a`` is ``(M, K)``; ``shape_b_t`` is ``(N, K)`` (W stored output-major,
    matching the FC convention -- see the module docstring). Both must be rank-2
    and agree on K; that agreement is asserted here so no kernel has to
    re-derive it, and a caller passing inconsistent shapes gets one clear error
    rather than sixteen confusing refusals.

    Raises:
        ValueError: if either shape is not rank-2, or the two disagree on K.
    """
    a = tuple(int(d) for d in shape_a)
    b = tuple(int(d) for d in shape_b_t)
    if len(a) != 2:
        raise ValueError(f"shape_a must be rank-2 (M, K); got {a}")
    if len(b) != 2:
        raise ValueError(f"shape_b_t must be rank-2 (N, K); got {b}")
    m, k = a
    n, k_b = b
    if k != k_b:
        raise ValueError(
            f"shape_a's K ({k}) does not match shape_b_t's K ({k_b}): "
            f"shape_a={a}, shape_b_t={b}"
        )
    bundle = ShapeBundle.of(a=a, b=b).with_shapes(derived={"output": (m, n)})
    return MatmulQuery(m=m, k=k, n=n, bundle=bundle)


def positive_dims(q: MatmulQuery) -> str | None:
    """Return a refusal reason if the problem has a non-positive extent."""
    if q.m < 1:
        return f"m ({q.m}) must be >= 1"
    if q.k < 1:
        return f"k ({q.k}) must be >= 1"
    if q.n < 1:
        return f"n ({q.n}) must be >= 1"
    return None
