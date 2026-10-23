"""``y = x``: data movement only (E07-04)."""

from __future__ import annotations

import torch

__all__ = ["identity"]


def identity(x: torch.Tensor, **_params: object) -> dict[str, torch.Tensor]:
    """``y = x``."""
    return {"y": x.clone()}
