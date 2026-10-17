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

One module per op (``ops/<op>.py``), its function re-exported here under the
op's name, which is what ``op = "<name>"`` looks up. PyTorch is test-only:
these modules import it, and only a comparison against a reference imports
this package. E07 adds an op's module as it migrates the family that needs
it.
"""

from __future__ import annotations

from ipu_kernels.reference.ops.concat import concat
from ipu_kernels.reference.ops.depth_to_space import depth_to_space
from ipu_kernels.reference.ops.identity import identity
from ipu_kernels.reference.ops.layernorm import layernorm
from ipu_kernels.reference.ops.linear import linear
from ipu_kernels.reference.ops.matmul import matmul
from ipu_kernels.reference.ops.matmul_nt import matmul_nt
from ipu_kernels.reference.ops.silu import silu
from ipu_kernels.reference.ops.softmax import softmax

__all__ = [
    "concat",
    "depth_to_space",
    "identity",
    "layernorm",
    "linear",
    "matmul",
    "matmul_nt",
    "silu",
    "softmax",
]
