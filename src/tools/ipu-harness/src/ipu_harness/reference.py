"""The PyTorch reference a kernel's outputs are compared with (E06-09, Q74b).

``[golden]`` names it: ``reference = "reference.py:fn"`` (a function in the
kernel folder, loaded from its file), or ``op = "layernorm"`` (the shared
``ipu_kernels.reference.ops.layernorm``, imported by name: Q62), or neither
with a ``reference.py`` in the folder (its ``reference``, which the loader
already fills in). Without any, the kernel is checked only against user
``expected`` files.

PyTorch is a test-only dependency: :func:`evaluate` imports it when it runs,
so loading a kernel, or running one against ``expected`` files, needs no torch.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from ipu_harness._codec import Array
from ipu_harness._plugins import import_named, load_file
from ipu_harness.manifest import KernelManifest
from ipu_harness.params import PlanError, Value

__all__ = ["ReferenceFn", "evaluate", "load_reference"]

#: ``fn(**inputs, **params) -> {output name: tensor}``.
ReferenceFn = Callable[..., Mapping[str, Any]]

#: Where the shared references live, by operation name.
SHARED = "ipu_kernels.reference.ops"


def _callable(found: object, what: str, manifest: KernelManifest, key: str) -> ReferenceFn:
    if not callable(found):
        raise PlanError(manifest, key, f"{what} is not a function")
    return found


def load_reference(manifest: KernelManifest, *, shared: str = SHARED) -> ReferenceFn | None:
    """The kernel's reference function, if it names one.

    Args:
        manifest: The kernel.
        shared: The module of shared references (a test names another).

    Returns:
        The function, or None when the kernel has no reference.

    Raises:
        PlanError: The named file or op does not load or define a function,
            located at ``golden.reference`` or ``golden.op``.
    """
    golden = manifest.golden
    if golden.reference is not None:
        file, _, name = golden.reference.partition(":")
        path = manifest.directory / file
        try:
            module = load_file(path)
        except Exception as error:
            raise PlanError(
                manifest, "golden.reference", f"cannot load {file}: {type(error).__name__}: {error}"
            ) from error
        return _callable(
            getattr(module, name, None), golden.reference, manifest, "golden.reference"
        )
    if golden.op is not None:
        try:
            ops = import_named(shared)
        except Exception as error:
            raise PlanError(
                manifest, "golden.op", f"cannot import {shared}: {type(error).__name__}: {error}"
            ) from error
        if ops is None:
            raise PlanError(manifest, "golden.op", f"{shared} is not installed")
        return _callable(
            getattr(ops, golden.op, None), f"{shared}.{golden.op}", manifest, "golden.op"
        )
    return None


def evaluate(
    fn: ReferenceFn, inputs: Mapping[str, Array], params: Mapping[str, Value]
) -> dict[str, Array]:
    """``fn``'s outputs for one case.

    Args:
        fn: From :func:`load_reference`.
        inputs: Each input as the format holds it (``inputs_for``); passed as
            float64 torch tensors.
        params: The case's parameters, passed as keywords.

    Returns:
        Each output, by name, as a float64 array.

    Raises:
        TypeError: ``fn`` returned something other than a mapping of tensors
            or arrays by name.
    """
    import torch  # test-only (Q74b): only a comparison against a reference needs it

    tensors = {
        name: torch.from_numpy(np.asarray(a, dtype=np.float64)) for name, a in inputs.items()
    }
    made = fn(**tensors, **params)
    if not isinstance(made, Mapping):
        raise TypeError(f"the reference returned {type(made).__name__}, not outputs by name")
    out = {}
    for name, value in made.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        out[name] = np.asarray(value, dtype=np.float64)
    return out
