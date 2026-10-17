"""Routing of the matmul family (E07-07, E07-08): the ``ipu_apps`` matmul
router's facts on the kernel tree's registry and ``//kernels:query``.

The old query named ``shape_a=(M, K)`` and ``shape_b_t=(N, K)``; a query now
names ``m``, ``k`` and ``n``. Every entry is one exact shape, and no two
entries share one (with their activation).
"""

from __future__ import annotations

import json

import pytest

from ipu_harness import query
from ipu_harness.registry import Registry, kernel_tree

#: Each entry's (m, k, n, activation) and template.
ENTRIES = {
    "matmul_64x64x64": (64, 64, 64, "none", "matmul_small"),
    "matmul_128x64x64": (128, 64, 64, "none", "matmul_small"),
    "matmul_128x64x128": (128, 64, 128, "none", "matmul_small"),
    "matmul_128x128": (128, 128, 128, "none", "matmul_small"),
}


@pytest.fixture(scope="module")
def registry():
    loaded = Registry.load(kernel_tree(), families=["matmul"])
    assert not loaded.invalid, loaded.invalid
    return loaded


def test_the_entries_and_their_templates(registry):
    assert {m.name for m in registry.kernels("matmul")} == set(ENTRIES)
    for name, (*_, template) in ENTRIES.items():
        assert registry.kernel(name).directory.name == template
    assert {t for *_, t in ENTRIES.values()} <= set(registry.templates)


@pytest.mark.parametrize("name", sorted(ENTRIES))
def test_each_entry_is_routed_by_its_exact_shape(registry, name, capsys):
    m, k, n, activation, _ = ENTRIES[name]
    verdict = registry.resolve("matmul", m=m, k=k, n=n, activation=activation)
    assert verdict.kernel_name == name, verdict.reason
    assert verdict.alternatives == ()
    assert verdict.label == f"//kernels/matmul:{name}"
    argv = ["matmul", f"m={m}", f"k={k}", f"n={n}", "--json"]
    assert query.main(argv, registry) == 0
    assert json.loads(capsys.readouterr().out)["app"] == name


@pytest.mark.parametrize("shape", [(64, 64, 32), (96, 64, 64), (128, 129, 128), (0, 64, 64)])
def test_other_shapes_are_refused(registry, shape):
    m, k, n = shape
    assert not registry.resolve("matmul", m=m, k=k, n=n)


def test_no_small_entry_fuses_silu(registry):
    assert not registry.resolve("matmul", m=128, k=128, n=128, activation="silu")
