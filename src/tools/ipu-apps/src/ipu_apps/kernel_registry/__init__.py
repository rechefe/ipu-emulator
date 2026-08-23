"""Op-agnostic kernel registry: which app implements a given computation.

Kernels declare themselves where they live. An app package exports a module
level ``SPEC`` (a :class:`~ipu_apps.kernel_registry.spec.KernelSpec`) saying
what operation it implements and which shapes it handles; discovery walks the
app tree and collects them. Nothing central lists the kernels, so adding one is
purely additive.

Query it at whichever level suits the caller::

    from ipu_apps.kernel_registry import resolve, lookup_layer

    # framework-native: a layer plus the shape it will see
    lookup_layer(nn.Softmax(dim=1), input_shape=(32, 300))

    # framework-free
    resolve("softmax", dim=1, shape=(32, 300))

Both return a :class:`~ipu_apps.kernel_registry.spec.Verdict` carrying the app
class, its constructor kwargs, the reasoning, and any caveats that still apply.

The design rules worth knowing before adding a kernel:

* a kernel's ``supports`` is the **only** statement of its domain -- app
  constructors delegate their guards to it rather than restating bounds;
* overlapping claims are resolved by declared ``cost``, never discovery order;
* coverage reports are **generated** by probing kernels, never hand-written;
* shapes travel as a role-keyed bundle (input/weight/bias/output), so
  multi-tensor operations fit without changing the query envelope.

See ``docs/content/adding-applications.md`` for the full contributor guide.
"""

from ipu_apps.kernel_registry.coverage import Boundary, boundaries, report
from ipu_apps.kernel_registry.discovery import Discovered, SkippedModule, discover
from ipu_apps.kernel_registry.layers import (
    UnsupportedLayer,
    adapters,
    from_layer,
    register_layer,
    softmax_bundle,
)
from ipu_apps.kernel_registry.registry import kernels, load, operations, resolve
from ipu_apps.kernel_registry.shapes import (
    BIAS,
    INPUT,
    OUTPUT,
    WEIGHT,
    MalformedQuery,
    ShapeBundle,
    flatten_to_matrix,
)
from ipu_apps.kernel_registry.spec import KernelSpec, Support, Verdict, no, yes


def lookup_layer(layer, input_shape, *, package: str = "ipu_apps") -> Verdict:
    """Resolve a framework layer plus the input shape it will receive.

    Args:
        layer:       A framework layer object (e.g. ``torch.nn.Softmax``).
            Matched by class name, so torch is never imported here.
        input_shape: Shape of the tensor the layer will be applied to.
        package:     Root package to search for kernels.

    Returns:
        A :class:`Verdict`; truthy when a kernel covers the query.

    Raises:
        UnsupportedLayer: if no adapter exists for this layer type, or the
            layer is configured in a way no kernel implements.
    """
    op, params = from_layer(layer, input_shape)
    return resolve(op, package=package, **params)


__all__ = [
    "BIAS",
    "Boundary",
    "Discovered",
    "INPUT",
    "KernelSpec",
    "MalformedQuery",
    "OUTPUT",
    "ShapeBundle",
    "SkippedModule",
    "Support",
    "UnsupportedLayer",
    "Verdict",
    "WEIGHT",
    "adapters",
    "boundaries",
    "discover",
    "flatten_to_matrix",
    "from_layer",
    "kernels",
    "load",
    "lookup_layer",
    "no",
    "operations",
    "register_layer",
    "report",
    "resolve",
    "softmax_bundle",
    "yes",
]
