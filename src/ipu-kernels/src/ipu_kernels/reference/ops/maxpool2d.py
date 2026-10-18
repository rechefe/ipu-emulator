"""2-D max-pooling of a ``[C, H, W]`` tensor (E07-14)."""

from __future__ import annotations

import torch

__all__ = ["maxpool2d"]


def maxpool2d(
    x: torch.Tensor, *, kernel_size: int, stride: int, padding: int, **_params: object
) -> dict[str, torch.Tensor]:
    """``torch.nn.functional.max_pool2d`` of ``x`` ``[C, H, W]`` over each
    channel: a square ``kernel_size`` window at ``stride``, the border
    ``padding`` wide and never the maximum (PyTorch pads with -inf); an
    incomplete last window is dropped (``ceil_mode=False``)."""
    y = torch.nn.functional.max_pool2d(x[None], kernel_size, stride, padding)
    return {"y": y[0]}
