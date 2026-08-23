"""Translating a framework layer into a registry query.

A PyTorch layer carries *configuration* (``dim``, ``in_channels``, ``stride``)
but never *shape*: ``nn.Softmax(dim=1)`` has no idea it will be handed a
32x300. Shape is what selects a kernel, so a layer alone cannot answer the
question -- :func:`from_layer` takes the layer and the input shape together.

Layers differ in how much they already know. ``nn.Linear`` and ``nn.Conv2d``
own their weight and bias, so those shapes (and the output shape) are
*derived*, not supplied. A bare ``a @ b`` has two independent inputs and no
layer object at all, so both must be given. Both funnel into the same
:class:`~ipu_apps.kernel_registry.shapes.ShapeBundle`.

Adapters are matched by class name rather than by ``isinstance``, which keeps
torch an optional dependency: the registry never imports it. Anything exposing
the right attributes works, including a stub in a test.

Two rules adapters follow, because both failure modes are silent and severe:

* **Enumerate what you understand; refuse the rest.** An adapter that ignores
  an attribute it does not model will happily answer for an operation the
  kernel does not implement.
* **Never assume a neighbour is equivalent.** ``LogSoftmax`` sits beside
  ``Softmax`` in torch and is a different function; it is refused unless a
  kernel actually claims it.
"""

from __future__ import annotations

from typing import Any, Callable

from ipu_apps.kernel_registry.shapes import Shape, ShapeBundle, flatten_to_matrix

# layer class name -> adapter(layer, input_shape) -> (op, params)
_ADAPTERS: dict[str, Callable[[Any, Shape], tuple[str, dict[str, Any]]]] = {}


class UnsupportedLayer(ValueError):
    """The layer, or the way it is configured, has no registry translation."""


def register_layer(*class_names: str):
    """Register an adapter for one or more framework layer class names.

    Adapters live next to the kernels they serve, so supporting a new layer
    type does not touch this module.
    """

    def decorate(fn):
        for name in class_names:
            _ADAPTERS[name] = fn
        return fn

    return decorate


def adapters() -> tuple[str, ...]:
    """Layer class names that currently have an adapter."""
    return tuple(sorted(_ADAPTERS))


def _require_attrs(layer: Any, *names: str) -> None:
    missing = [n for n in names if not hasattr(layer, n)]
    if missing:
        raise UnsupportedLayer(
            f"{type(layer).__name__} is missing expected attribute(s) "
            f"{', '.join(missing)}; it does not look like the layer this "
            f"adapter was written for"
        )


def from_layer(layer: Any, input_shape: Shape) -> tuple[str, dict[str, Any]]:
    """Translate ``layer`` + ``input_shape`` into ``(op, params)``.

    Raises:
        UnsupportedLayer: if no adapter is registered for this layer type.
    """
    name = type(layer).__name__
    adapter = _ADAPTERS.get(name)
    if adapter is None:
        known = ", ".join(adapters()) or "none"
        raise UnsupportedLayer(
            f"no adapter for layer type {name!r}. Layers with an adapter: "
            f"{known}. Add one with @register_layer({name!r}) next to the "
            f"kernel that implements it."
        )
    return adapter(layer, tuple(int(d) for d in input_shape))


# -- built-in adapters ------------------------------------------------------


@register_layer("Softmax")
def _softmax_layer(layer: Any, input_shape: Shape) -> tuple[str, dict[str, Any]]:
    """``nn.Softmax(dim=...)`` -> the ``softmax`` operation.

    ``nn.Softmax`` created without an explicit ``dim`` has ``dim=None``, which
    torch itself treats as deprecated and resolves with a heuristic. Rather
    than replicate that heuristic (and risk disagreeing with the framework on
    which axis is normalised), it is refused.
    """
    _require_attrs(layer, "dim")
    if layer.dim is None:
        raise UnsupportedLayer(
            "Softmax(dim=None) does not state which axis to normalise; torch "
            "resolves it with a deprecated heuristic. Construct the layer with "
            "an explicit dim."
        )
    return "softmax", {"dim": int(layer.dim), "shape": input_shape}


@register_layer("LogSoftmax", "Softmin")
def _unsupported_softmax_relatives(layer: Any, input_shape: Shape):
    """Refuse near-neighbours of Softmax explicitly.

    These sit beside ``Softmax`` in ``torch.nn`` and share its signature, so a
    permissive adapter would route them to a softmax kernel and return
    confidently wrong numbers.
    """
    name = type(layer).__name__
    detail = {
        "LogSoftmax": "computes log(softmax(x)), not softmax(x)",
        "Softmin": "computes softmax(-x), not softmax(x)",
    }[name]
    raise UnsupportedLayer(
        f"{name} {detail}; no kernel implements it. Using a softmax kernel "
        f"here would return confidently wrong values."
    )


def softmax_bundle(shape: Shape, dim: int) -> tuple[ShapeBundle, int, tuple[int, int]]:
    """Build the shape bundle for a softmax query.

    Softmax is shape-preserving, so the output shape is derived and equal to
    the input. A rank > 2 input is flattened around ``dim`` (recorded as a note
    on the bundle, never silently).

    Returns:
        ``(bundle, dim_2d, shape_2d)``.
    """
    shape_2d, dim_2d, note = flatten_to_matrix(shape, dim)
    bundle = ShapeBundle.of(input=shape).with_shapes(
        derived={"output": shape},
        notes=(note,) if note else (),
    )
    return bundle, dim_2d, shape_2d
