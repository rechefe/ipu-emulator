"""``y = a @ b`` (E06-09)."""

from __future__ import annotations

import torch

__all__ = ["matmul"]


def matmul(a: torch.Tensor, b: torch.Tensor, **_params: object) -> dict[str, torch.Tensor]:
    """``y = a @ b``."""
    return {"y": a @ b}
