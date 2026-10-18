"""``y = a + b`` (E07-06)."""

from __future__ import annotations

import torch

__all__ = ["residual_add"]


def residual_add(a: torch.Tensor, b: torch.Tensor, **_params: object) -> dict[str, torch.Tensor]:
    """``y = a + b``, elementwise."""
    return {"y": a + b}
