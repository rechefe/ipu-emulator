"""``y = a @ b`` (E06-09), or ``y = a @ b^T`` (E07-13)."""

from __future__ import annotations

import torch

__all__ = ["matmul"]


def matmul(
    a: torch.Tensor, b: torch.Tensor, *, transpose_b: bool = False, **_params: object
) -> dict[str, torch.Tensor]:
    """``y = a @ b``, batched over leading dimensions.

    Args:
        a: ``[..., m, k]``.
        b: ``[..., k, n]``, or ``[..., n, k]`` with ``transpose_b``.
        transpose_b: Contract ``b``'s last dimension: a kernel parameter of
            that name, for a kernel whose ``b`` is stored row per output
            column (``attn_v``'s query-major P).

    Returns:
        ``{"y": a @ b}`` (``a @ b^T`` with ``transpose_b``).
    """
    return {"y": a @ (b.transpose(-1, -2) if transpose_b else b)}
