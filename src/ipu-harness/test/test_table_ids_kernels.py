"""No kernel relies on the value of a table ID.

The default case of every kernel of the tree runs twice in its first format:
with the IDs the binder gives by default (the lowest free, in ``[abi]`` order)
and with the free IDs in a seeded random order. A program that read an ID as a
number, or kept it from one table to another, would give other outputs or
cycles, or fault. Heavy tier (nightly): it repeats the work of every kernel's
own tests; the generated tests run only the default IDs.
"""

from __future__ import annotations

import pytest

from ipu_harness.cases import expand, rng
from ipu_harness.registry import Registry, kernel_tree
from ipu_harness.runner import RunOptions, run_kernel
from ipu_harness.table_ids import MAX_TABLES

_REGISTRY = Registry.load(kernel_tree())

KERNELS = [pytest.param(m, id=name) for name, m in sorted(_REGISTRY.manifests.items())]


@pytest.mark.parametrize("manifest", KERNELS)
def test_the_default_case_runs_the_same_whatever_the_free_ids(manifest):
    case = next(c for c in expand(manifest) if c.name == "default")
    fmt = (case.formats or manifest.formats)[0]
    plain = run_kernel(manifest, case, fmt)
    assert plain.passed, plain.describe()

    order = [int(i) for i in rng(manifest.name, case.name, "table ids").permutation(MAX_TABLES)]
    shuffled = run_kernel(manifest, case, fmt, RunOptions(table_id_order=tuple(order)))
    assert shuffled.passed, f"{shuffled.describe()} (ID order {order})"
    assert shuffled.raw_outputs == plain.raw_outputs
    assert shuffled.cycles == plain.cycles
