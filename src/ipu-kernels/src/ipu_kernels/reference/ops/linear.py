"""``y = activation(a @ b)``: a projection (E07-02)."""

from __future__ import annotations

import torch

__all__ = ["linear"]


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
