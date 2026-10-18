"""``y = x * sigmoid(x)`` (E06-09)."""

from __future__ import annotations

import torch

__all__ = ["silu"]


def silu(x: torch.Tensor, **_params: object) -> dict[str, torch.Tensor]:
    """``y = x * sigmoid(x)``."""
    return {"y": torch.nn.functional.silu(x)}
