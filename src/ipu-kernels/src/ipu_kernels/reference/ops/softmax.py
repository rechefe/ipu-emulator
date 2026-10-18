"""Softmax along ``dim`` (E06-09)."""

from __future__ import annotations

import torch

__all__ = ["softmax"]


def softmax(x: torch.Tensor, *, dim: int, **_params: object) -> dict[str, torch.Tensor]:
    """Softmax along ``dim``."""
    return {"y": torch.softmax(x, dim=dim)}
