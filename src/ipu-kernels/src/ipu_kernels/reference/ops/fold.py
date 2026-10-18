"""Fold: the ``[c, h, w]`` tensor whose stride-2 phases are given (E07-05)."""

from __future__ import annotations

import torch

__all__ = ["fold"]


def fold(x: torch.Tensor, *, h: int, w: int, **_params: object) -> dict[str, torch.Tensor]:
    """The inverse of ``unfold``: ``[4, c, (h / 2) * (w / 2)] -> [c, h, w]``,
    ``y[:, i::2, j::2] = x[2 * i + j]`` (``torch.nn.functional.fold`` with a
    2 x 2 kernel at stride 2)."""
    cols = x.permute(1, 0, 2)  # [c, 4, L]
    return {"y": torch.nn.functional.fold(cols, (h, w), kernel_size=2, stride=2)[:, 0]}
