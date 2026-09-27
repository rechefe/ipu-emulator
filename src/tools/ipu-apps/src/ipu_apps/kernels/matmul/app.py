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

from ipu_apps.kernel_registry import ExecutionConfig, ShapeBundle, folder_spec, kernel_folder, no, yes

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


def matmul_spec(app_class, *, m: int, k: int, n: int, role: str, activation: str = "none"):
    """KernelSpec for a fixed ``(m, k, n)`` matmul kernel.

    ``role`` names where the kernel sits in the network, for ``explain``.
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
    )
