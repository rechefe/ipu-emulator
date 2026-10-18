"""Depth to space: a pixel shuffle to one output channel (E07-04)."""

from __future__ import annotations

import torch

__all__ = ["depth_to_space"]


def depth_to_space(
    x: torch.Tensor, *, upscale_factor: int, **_params: object
) -> dict[str, torch.Tensor]:
    """``[r * r, h, w] -> [h * r, w * r]``: ``torch.nn.functional.pixel_shuffle``
    with ``r = upscale_factor`` (``y[i * r + a, j * r + b] = x[a * r + b, i, j]``)."""
    return {"y": torch.nn.functional.pixel_shuffle(x[None], upscale_factor)[0, 0]}
