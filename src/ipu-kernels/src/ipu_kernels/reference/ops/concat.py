"""``y = cat((a, b), dim=0)`` (E07-04)."""

from __future__ import annotations

import torch

__all__ = ["concat"]


def concat(a: torch.Tensor, b: torch.Tensor, **_params: object) -> dict[str, torch.Tensor]:
    """``a``'s rows, then ``b``'s: the channel concat with one channel per row."""
    return {"y": torch.cat((a, b), dim=0)}
