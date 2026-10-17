"""LayerNorm along ``dim``, with no eps (E06-09, Q57)."""

from __future__ import annotations

import torch

__all__ = ["layernorm"]


def layernorm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    *,
    dim: int,
    **_params: object,
) -> dict[str, torch.Tensor]:
    """LayerNorm along ``dim``, with no eps (Q57): what the kernels compute.

    ``y = gamma * (x - mean) * rsqrt(var) + beta``, with mean and the biased
    variance taken along ``dim`` and ``gamma``, ``beta`` indexed along it. A
    zero variance (a constant slice) gives ``rsqrt = 0``, as the kernel's rsqrt
    activation maps a non-positive input to 0, so the slice becomes ``beta``.
    Whether AaQ keeps that, or adds an eps, is open (Q57).
    """
    mean = x.mean(dim=dim, keepdim=True)
    centered = x - mean
    var = (centered * centered).mean(dim=dim, keepdim=True)
    inv_std = torch.where(var > 0, torch.rsqrt(var), torch.zeros_like(var))
    shape = [1] * x.dim()
    shape[dim] = -1
    return {"y": gamma.reshape(shape) * (centered * inv_std) + beta.reshape(shape)}
