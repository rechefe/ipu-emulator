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

This module holds only the *mechanism*. Every adapter lives beside the kernels
it serves (softmax's are in :mod:`ipu_apps.softmax._spec_support`), so no
operation's vocabulary leaks into the op-agnostic core, and supporting a new
layer type never touches this file. :func:`from_layer` runs discovery before
looking an adapter up, because a kernel package that has not been imported has
not registered its adapters yet.

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


def from_layer(
    layer: Any, input_shape: Shape, *, package: str = "ipu_apps"
) -> tuple[str, dict[str, Any]]:
    """Translate ``layer`` + ``input_shape`` into ``(op, params)``.

    Args:
        layer:       A framework layer object, matched by class name.
        input_shape: Shape of the tensor the layer will be applied to.
        package:     Root package to discover adapters in. Discovery runs
            first: adapters register as a side effect of importing the kernel
            package that declares them, so without this an adapter declared
            beside its kernel -- the documented way to add one -- would be
            invisible until something else happened to import it.

    Raises:
        UnsupportedLayer: if no adapter is registered for this layer type.
    """
    # Imported here rather than at module scope: registry imports discovery,
    # which imports the app tree, which imports this module.
    from ipu_apps.kernel_registry.registry import load

    load(package)
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
