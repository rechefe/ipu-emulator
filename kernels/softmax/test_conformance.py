"""The softmax conformance suite (E07-11): whatever kernel the registry picks
for a shape must compute softmax there.

For each (shape, dim), the registry resolves the query, and the kernel it
chose runs that shape (its default case at the query's values, the data drawn
as the ``ipu_apps`` suite drew it) and is compared with NumPy. A kernel whose
domain claims a shape it mishandles fails here, not in a user's hands.
"""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pytest

from ipu_harness.assembly import AssemblyCache
from ipu_harness.cases import expand
from ipu_harness.registry import default_registry
from ipu_harness.runner import run_kernel
from ipu_isa.formats import ElementFormat

SHAPES = [
    (6, 8),
    (6, 20),
    (6, 64),
    (5, 100),
    (4, 128),
    (3, 129),
    (3, 200),
    (2, 256),
    (7, 300),
    (9, 16),
    (12, 32),
    (33, 64),
    (17, 65),
    (10, 127),
]

CACHE = AssemblyCache()


@pytest.fixture(scope="module")
def registry():
    loaded = default_registry()
    assert not loaded.invalid, loaded.invalid
    return loaded


def reference(x: np.ndarray, axis: int) -> np.ndarray:
    z = np.exp(x - x.max(axis=axis, keepdims=True))
    return z / z.sum(axis=axis, keepdims=True)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dim", [0, 1])
def test_the_resolved_kernel_computes_softmax(registry, shape, dim):
    rows, cols = shape
    length = {"n": cols} if dim == 1 else {"width": cols}
    verdict = registry.resolve("softmax", rows=rows, dim=dim, **length)
    assert verdict, verdict.reason
    kernel = verdict.kernel
    # The default case at the query's values, x drawn as the ipu_apps suite did.
    default = next(c for c in expand(kernel) if c.name == "default")
    values = MappingProxyType(
        {**default.values, **verdict.params, "seed": sum(shape) + dim, "scale": 3.0}
    )
    empty = MappingProxyType({})
    case = replace(
        default,
        name=f"conformance[{rows}x{cols},dim={dim}]",
        values=values,
        inputs=empty,
        expected=empty,
        decl=replace(default.decl, values=values, inputs=empty, expected=empty),
    )
    result = run_kernel(kernel, case, ElementFormat.FP32, cache=CACHE)
    assert result.passed, result.describe()
    x = (np.random.RandomState(sum(shape) + dim).randn(*shape) * 3.0).astype(np.float32)
    y = np.asarray(result.outputs["y"])
    assert y.shape == shape, f"{kernel.name} wrote {y.shape} for {shape}"
    assert np.abs(y - reference(x, dim)).max() < 1e-4
