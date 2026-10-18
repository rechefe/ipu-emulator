"""Keypoint selection gate and soft survivor count (E07-15)."""

from __future__ import annotations

import torch

__all__ = ["score_threshold"]


def score_threshold(
    x: torch.Tensor, *, threshold: float, temperature: float, **_params: object
) -> dict[str, torch.Tensor]:
    """``x`` ``[count]`` scores: ``y[:count]`` is ``relu(x - threshold)`` (the
    exact set ``x > threshold`` keeps a positive value), and ``y[count]`` is
    ``sum(sigmoid(temperature * (x - threshold)))``, the soft count a host
    bisects ``threshold`` on."""
    shifted = x - threshold
    count = torch.sigmoid(temperature * shifted).sum()
    return {"y": torch.cat((torch.relu(shifted), count[None]))}
