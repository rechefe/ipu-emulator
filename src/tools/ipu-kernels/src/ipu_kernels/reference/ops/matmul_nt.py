"""``y = a @ b^T`` (E07-13): a matmul whose ``b`` is stored a row per output column."""

from __future__ import annotations

import torch

__all__ = ["matmul_nt"]


def matmul_nt(a: torch.Tensor, b: torch.Tensor, **_params: object) -> dict[str, torch.Tensor]:
    """``y = a @ b^T``, batched over leading dimensions ("NT": ``a`` as is,
    ``b`` transposed, as BLAS names it), for ``attn_v``'s query-major P.

    Args:
        a: ``[..., m, k]``.
        b: ``[..., n, k]``.

    Returns:
        ``{"y": a @ b^T}``.
    """
    return {"y": a @ b.transpose(-1, -2)}
