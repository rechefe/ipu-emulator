"""Unfold: the four stride-2 phases of a ``[c, h, w]`` tensor (E07-05)."""

from __future__ import annotations

import torch

__all__ = ["unfold"]


def unfold(x: torch.Tensor, **_params: object) -> dict[str, torch.Tensor]:
    """``y[2 * i + j] = x[:, i::2, j::2]``, each phase flattened per channel:
    ``[c, h, w] -> [4, c, (h / 2) * (w / 2)]``.

    That is ``torch.nn.functional.unfold`` with a 2 x 2 kernel at stride 2,
    its ``[c, 4, L]`` columns permuted to ``[4, c, L]`` (phase-major, as the
    kernels store the four streams).
    """
    c = x.shape[0]
    cols = torch.nn.functional.unfold(x[:, None], kernel_size=2, stride=2)  # [c, 4, L]
    return {"y": cols.reshape(c, 4, -1).permute(1, 0, 2).contiguous()}
