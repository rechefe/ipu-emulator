"""One shared reference per operation, in PyTorch (E06-09, Q74b).

A kernel selects one with ``[golden] op = "<name>"``. ``ipu_harness.reference``
calls it with the case's inputs as float64 tensors, by name, and the case's
parameters as keywords. So an op names its inputs (``x``, ``a``, ``b``,
``gamma``, ``beta``) and the kernel declares its tensors by those names. An op
returns one output, ``y``, so a kernel that uses ``op`` has exactly one output,
named ``y``. Parameters an op does not read (the kernel's geometry) are
accepted and ignored. An op that reduces along an axis takes ``dim`` with no
default: kernel layouts differ (the normalize kernels keep channels on axis 0),
so the kernel declares ``dim`` in ``[params]``, and a missing one is an error,
not a reduction along the wrong axis.

PyTorch is test-only: this module imports it, and only a comparison against a
reference imports this module. E07 adds ``maxpool2d`` and ``residual_add`` as
it migrates the families (``conv2d`` came with the convolutions, E07-16).
"""

from __future__ import annotations

from typing import Any

import torch

__all__ = ["conv2d", "layernorm", "linear", "matmul", "silu", "softmax"]


def matmul(a: torch.Tensor, b: torch.Tensor, **_params: object) -> dict[str, torch.Tensor]:
    """``y = a @ b``."""
    return {"y": a @ b}


#: The activations :func:`linear` applies, by the name a kernel's ``activation``
#: parameter gives.
_ACTIVATIONS = {"none": lambda y: y, "silu": torch.nn.functional.silu}


def linear(
    a: torch.Tensor, b: torch.Tensor, *, activation: str = "none", **_params: object
) -> dict[str, torch.Tensor]:
    """``y = activation(a @ b)``: a projection, with ``activation`` ``"none"``
    or ``"silu"`` (a kernel's ``activation`` parameter, when it declares one).

    ``a`` is the output-major weights ``[n_out, k]`` and ``b`` the
    channel-major data ``[..., k, n_tok]``, so ``y`` is ``[..., n_out, n_tok]``
    (``@`` broadcasts the weights over the leading axes).

    Raises:
        ValueError: ``activation`` is neither.
    """
    if activation not in _ACTIVATIONS:
        raise ValueError(f"linear: activation {activation!r} is not one of {sorted(_ACTIVATIONS)}")
    return {"y": _ACTIVATIONS[activation](a @ b)}


def silu(x: torch.Tensor, **_params: object) -> dict[str, torch.Tensor]:
    """``y = x * sigmoid(x)``."""
    return {"y": torch.nn.functional.silu(x)}


def softmax(x: torch.Tensor, *, dim: int, **_params: object) -> dict[str, torch.Tensor]:
    """Softmax along ``dim``."""
    return {"y": torch.softmax(x, dim=dim)}


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


#: The geometry :func:`conv2d` reads from a kernel's parameters, with torch's defaults.
_CONV_GEOMETRY = {"stride": 1, "padding": 0, "dilation": 1, "groups": 1}


def conv2d(
    x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None = None, **params: Any
) -> dict[str, torch.Tensor]:
    """``y = act(conv2d(x, w) + b)`` of one image: ``x`` is ``[in, height,
    width]``, ``w`` ``[out, in / groups, k, k]`` (torch's layout) and ``b``
    ``[out]``, or absent. ``stride``, ``padding``, ``dilation`` and ``groups``
    are the kernel's parameters (torch's defaults when it has none). The two
    conv2d vocabularies name the activation differently, so either selects
    ReLU: ``activation`` (``"none"`` or ``"relu"``, the tiled kernels') or
    ``apply_relu`` (the universal ones').

    Raises:
        ValueError: ``activation`` is neither.
    """
    activation = params.get("activation", "none")
    if activation not in ("none", "relu"):
        raise ValueError(f"conv2d: activation {activation!r} is not one of ['none', 'relu']")
    geometry = {name: int(params.get(name, default)) for name, default in _CONV_GEOMETRY.items()}
    y = torch.nn.functional.conv2d(x[None], w, b, **geometry)[0]
    relu = bool(params.get("apply_relu", False)) or activation == "relu"
    return {"y": torch.relu(y) if relu else y}
