"""``y = act(conv2d(x, w) + b)`` of one image (E07-16)."""

from __future__ import annotations

from typing import Any

import torch

__all__ = ["conv2d"]

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
