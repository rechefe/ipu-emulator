"""Per-cell detector confidence and its threshold gate (E07-15)."""

from __future__ import annotations

import torch

__all__ = ["channel_peak"]


def channel_peak(
    x: torch.Tensor, *, threshold: float, **_params: object
) -> dict[str, torch.Tensor]:
    """``x`` ``[channels, cells]``: ``y[0]`` is each cell's maximum over the
    channels (the confidence; its argmax is the softmax's), ``y[1]`` is
    ``relu(y[0] - threshold)`` (the gate)."""
    confidence = x.max(dim=0).values
    return {"y": torch.stack((confidence, torch.relu(confidence - threshold)))}
