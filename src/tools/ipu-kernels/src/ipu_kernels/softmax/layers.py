"""The softmax family's layer adapters (E07-11), for ``Registry.from_layer``.

``nn.Softmax(dim=d)`` applied to a tensor of any shape becomes the query the
kernels route on: ``softmax`` with ``rows``, ``dim`` and the row length, ``n``
along each row (``dim = 1``) or ``width`` down each column (``dim = 0``).
torch's ``dim`` convention holds: a negative dim counts from the end, and a
1-D tensor is one row. A rank > 2 tensor is a batch of 2-D problems when the
softmax axis is its first or last: the other axes flatten into one. An interior
axis would need the others transposed around it, which reinterprets the
caller's memory, so it is refused.

``LogSoftmax`` and ``Softmin`` sit beside ``Softmax`` in ``torch.nn`` with the
same signature; a permissive adapter would route them to a softmax kernel and
return confidently wrong values, so they are refused by name.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from ipu_harness.registry import LayerAdapter, UnsupportedLayer

__all__ = ["LAYER_ADAPTERS", "softmax_query"]


def softmax_query(shape: tuple[int, ...], dim: int) -> dict[str, int]:
    """The softmax query of a ``shape`` tensor normalised along ``dim``.

    Args:
        shape: The tensor's shape, rank >= 1.
        dim: The axis, torch's convention (negative counts from the end).

    Returns:
        ``{"rows", "dim", "n"}`` (along rows) or ``{"rows", "dim", "width"}``
        (down columns), for :meth:`Registry.resolve`.

    Raises:
        UnsupportedLayer: ``dim`` is out of range, or an interior axis.
    """
    rank = len(shape)
    if not -rank <= dim < rank:
        raise UnsupportedLayer(f"dim {dim} is out of range for a rank-{rank} shape {shape}")
    dim %= rank
    if rank == 1:
        return {"rows": 1, "n": shape[0], "dim": 1}
    others = math.prod(d for i, d in enumerate(shape) if i != dim)
    if dim == rank - 1:
        return {"rows": others, "n": shape[dim], "dim": 1}
    if dim == 0:
        return {"rows": shape[0], "width": others, "dim": 0}
    raise UnsupportedLayer(
        f"dim {dim} is an interior axis of a rank-{rank} shape {shape}; flattening around "
        "it would transpose the other axes, which reinterprets the caller's memory. Move "
        "the softmax axis to the first or last dimension first (permute or movedim)."
    )


def _softmax(layer: object, shape: tuple[int, ...]) -> tuple[str, Mapping[str, object]]:
    """``nn.Softmax(dim=...)``: the ``softmax`` query of its input.

    Raises:
        UnsupportedLayer: The layer has no ``dim``, or ``dim=None`` (torch
            resolves that with a deprecated heuristic, which this does not
            replicate), or the dim does not flatten (:func:`softmax_query`).
    """
    if not hasattr(layer, "dim"):
        raise UnsupportedLayer(
            f"{type(layer).__name__} has no dim attribute; it does not look like the "
            "layer this adapter was written for"
        )
    if layer.dim is None:
        raise UnsupportedLayer(
            "Softmax(dim=None) does not state which axis to normalise; torch resolves it "
            "with a deprecated heuristic. Construct the layer with an explicit dim."
        )
    return "softmax", softmax_query(shape, int(layer.dim))


def _relative(layer: object, shape: tuple[int, ...]) -> tuple[str, Mapping[str, object]]:
    """``LogSoftmax`` and ``Softmin``: refused, never routed to a softmax kernel.

    Raises:
        UnsupportedLayer: Always.
    """
    name = type(layer).__name__
    computes = {
        "LogSoftmax": "computes log(softmax(x)), not softmax(x)",
        "Softmin": "computes softmax(-x), not softmax(x)",
    }[name]
    raise UnsupportedLayer(
        f"{name} {computes}; no kernel implements it, and a softmax kernel here would "
        "return confidently wrong values"
    )


#: The adapters, by layer class name.
LAYER_ADAPTERS: Mapping[str, LayerAdapter] = {
    "Softmax": _softmax,
    "LogSoftmax": _relative,
    "Softmin": _relative,
}
