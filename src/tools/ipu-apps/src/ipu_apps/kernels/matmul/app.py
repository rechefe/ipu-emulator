"""Shared code for the fixed-shape matmul kernels: query vocabulary and SPEC helper.

All matmul kernels answer the same query -- two 2-D operand shapes, plus the
activation fused into the output store -- so the parameter unpacking and the
vocabulary they reason about (``m``, ``k``, ``n``) live here once.

Every kernel computes ``C = act(A @ W^T)``:

``shape_a``    the input matrix's shape, ``(M, K)``
``shape_b_t``  the weight matrix's shape *as stored*, ``(N, K)`` -- W is kept
               output-major (row ``n`` holds the K inputs feeding output n),
               the same convention the fully-connected layer uses, so no
               transpose crosses the registry boundary
``activation`` optional, ``"none"`` (default) or ``"silu"``. The FFN1
               expansion kernels fuse ``silu`` into their store; routing a
               plain matmul to one of them would return confidently wrong
               values, so the activation is part of the query.

Unlike softmax's kernels, which cover a *range* of row counts, every matmul
kernel is a single fixed (M, K, N) triple with no padding or chunking
tolerance -- the match is exact or refused.

Two file layouts exist, and each kernel's ``cases.py`` states its own:

* **row-major** (``matmul_64x64x64``, ``matmul_128x*``): the input file is A,
  ``(M, K)``; the output file is dense C, ``(M, N)``.
* **channel-major** (``matmul_*_x128``): the input file is D = A^T,
  ``(K, M)``, as the transformer blocks hold it; the output file is raw XMEM
  rows, ``(N_TG, N, 128)``, with the first ``N_TOK`` lanes of each row valid.
"""

from __future__ import annotations

from dataclasses import dataclass

from ipu_apps.kernel_registry import ExecutionConfig, ShapeBundle, Table, folder_spec, kernel_folder, no, yes

OP = "matmul"
ACTIVATIONS = ("none", "silu")


@dataclass(frozen=True)
class MatmulQuery:
    """A matmul query reduced to what the kernels route on."""

    m: int
    k: int
    n: int
    activation: str
    bundle: ShapeBundle

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.m, self.k, self.n)


def matmul_query(shape_a, shape_b_t, activation: str = "none") -> MatmulQuery:
    """Normalise a matmul query into what every matmul kernel routes on.

    Raises:
        ValueError: if either shape is not rank-2, the two disagree on K, or
            the activation is unknown.
    """
    a = tuple(int(d) for d in shape_a)
    b = tuple(int(d) for d in shape_b_t)
    if len(a) != 2:
        raise ValueError(f"shape_a must be rank-2 (M, K); got {a}")
    if len(b) != 2:
        raise ValueError(f"shape_b_t must be rank-2 (N, K); got {b}")
    if activation not in ACTIVATIONS:
        raise ValueError(f"activation must be one of {ACTIVATIONS}; got {activation!r}")
    m, k = a
    n, k_b = b
    if k != k_b:
        raise ValueError(
            f"shape_a's K ({k}) does not match shape_b_t's K ({k_b}): "
            f"shape_a={a}, shape_b_t={b}"
        )
    bundle = ShapeBundle.of(a=a, b=b).with_shapes(derived={"output": (m, n)})
    return MatmulQuery(m=m, k=k, n=n, activation=activation, bundle=bundle)


def _query(params) -> MatmulQuery:
    return matmul_query(params["shape_a"], params["shape_b_t"],
                        params.get("activation", "none"))


def positive_dims(q: MatmulQuery) -> str | None:
    """Return a refusal reason if the problem has a non-positive extent."""
    for name in ("m", "k", "n"):
        if getattr(q, name) < 1:
            return f"{name} ({getattr(q, name)}) must be >= 1"
    return None


def matmul_spec(app_class, *, m: int, k: int, n: int, role: str, activation: str = "none",
                tables=None):
    """KernelSpec for a fixed ``(m, k, n)`` matmul kernel.

    ``role`` names where the kernel sits in the network, for ``explain``.
    ``tables`` is the kernel's ``SPEC.tables`` declaration, when it has one.
    """
    name = kernel_folder(app_class)

    def supports(**params):
        q = _query(params)
        reason = positive_dims(q)
        if reason is None and q.shape != (m, k, n):
            reason = f"handles exactly (M, K, N) == {(m, k, n)}; got {q.shape}"
        if reason is None and q.activation != activation:
            reason = f"fuses activation={activation!r} into its store; query asks for {q.activation!r}"
        return no(reason) if reason else yes()

    act = "" if activation == "none" else f" with fused {activation}"
    return folder_spec(
        app_class,
        op=OP,
        variant=name.removeprefix("matmul_"),
        requires=("shape_a", "shape_b_t"),
        tags=("fp32-wide",),
        supports=supports,
        build=lambda **params: {},
        explain=lambda **params: (
            f"(M, K, N) == {(m, k, n)} exactly: the fixed-shape {role} kernel{act}."),
        bundle=lambda **params: _query(params).bundle,
        # Exact-shape match: no two matmul kernels share a (triple, activation).
        cost=lambda **params: 0.0,
        execution=ExecutionConfig(mode="fp32"),
        tables=tables,
    )


def streamed_weights_tables(*, m: int, k: int) -> dict[str, Table]:
    """XMEM tables of the one-row-per-vector matmuls (``matmul_128x128`` style).

    A (one row per output row) and the output are walked once. T = W^T (one
    row per k) is re-read in full for every row of A, and the prefetch of
    T[k+1] on the last k reads one row past it (its data is not used): that
    row is declared and allocated as overhead.
    """
    return {
        "input": Table(cr=12, rows=m),
        "weights": Table(cr=11, repetitions=m, rows=k, overhead_rows=1),
        "output": Table(cr=2, rows=m),
    }


def streamed_data_tables(*, n_out: int, data_rows: int, data_stride: int, walks: int,
                         chunks: int, outputs: dict[str, int], reread_chunks: bool = False,
                         ) -> dict[str, Table]:
    """XMEM tables of the ``*_x128`` matmuls.

    The data D (``data_rows`` rows, ``data_stride`` per k) is re-walked
    ``walks`` times -- once per output feature (and token group) -- and the
    loop's next-row load runs one stride past it. W is one table through CR9:
    feature ``j``'s 128-element chunk ``c`` is row ``j * chunks + c``, walked
    once in order. With ``reread_chunks`` (a second token group reloading
    the feature's chunks) each feature steps back ``chunks - 1`` rows. Every
    output table holds ``n_out`` rows.
    """
    tables = {
        "data": Table(cr=12, rows=data_rows, repetitions=walks, overhead_rows=data_stride),
        "weights": Table(cr=9, rows=n_out * chunks,
                         jump_back=chunks - 1 if reread_chunks else 0),
    }
    for name, cr in outputs.items():
        tables[name] = Table(cr=cr, rows=n_out)
    return tables
