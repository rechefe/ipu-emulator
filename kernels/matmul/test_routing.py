"""Routing of the matmul family (E07-07, E07-08): the ``ipu_apps`` matmul
router's facts on the kernel tree's registry and ``//kernels:query``.

The old query named ``shape_a=(M, K)`` and ``shape_b_t=(N, K)``; a query now
names ``m``, ``k`` and ``n`` (and ``activation`` for an FFN1 shape). Every
entry is one exact shape, and no two entries share one with their
activation: a plain query at an FFN1 shape is refused, as before.
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
    "matmul_144x144_x128": (256, 144, 144, "none", "matmul_x128_2tg_c2"),
    "matmul_432x144_x128": (256, 144, 432, "none", "matmul_x128_2tg_c2"),
    "matmul_288x144_x128": (256, 144, 288, "silu", "matmul_x128_2tg_c2"),
    "matmul_144x288_x128": (256, 288, 144, "none", "matmul_x128_2tg_c3"),
    "matmul_192x192_x128": (64, 192, 192, "none", "matmul_x128_c2"),
    "matmul_384x192_x128": (64, 192, 384, "silu", "matmul_x128_c2"),
    "matmul_240x240_x128": (16, 240, 240, "none", "matmul_x128_c2"),
    "matmul_480x240_x128": (16, 240, 480, "silu", "matmul_x128_c2"),
    "matmul_720x240_x128": (16, 240, 720, "none", "matmul_x128_c2"),
    "matmul_576x192_x128": (64, 192, 576, "none", "matmul_x128_c2_qkv"),
    "matmul_192x384_x128": (64, 384, 192, "none", "matmul_x128_c3"),
    "matmul_240x480_x128": (16, 480, 240, "none", "matmul_x128_c4"),
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
    argv = ["matmul", f"m={m}", f"k={k}", f"n={n}", f"activation={activation}", "--json"]
    assert query.main(argv, registry) == 0
    assert json.loads(capsys.readouterr().out)["app"] == name


@pytest.mark.parametrize("shape", [(64, 64, 32), (96, 64, 64), (128, 129, 128), (0, 64, 64)])
def test_other_shapes_are_refused(registry, shape):
    m, k, n = shape
    assert not registry.resolve("matmul", m=m, k=k, n=n)


def test_no_small_entry_fuses_silu(registry):
    assert not registry.resolve("matmul", m=128, k=128, n=128, activation="silu")


def test_entries_are_7_templates():
    assert len({t for *_, t in ENTRIES.values()}) == 7


@pytest.mark.parametrize("name", sorted(n for n, e in ENTRIES.items() if e[3] == "silu"))
def test_a_silu_shape_routes_only_with_activation(registry, name):
    """A plain matmul at an FFN1 shape must not reach the SiLU-fused kernel
    (the old router's test_matmul_silu_shape_routes_only_with_activation)."""
    m, k, n, *_ = ENTRIES[name]
    assert not registry.resolve("matmul", m=m, k=k, n=n)
    assert not registry.resolve("matmul", m=m, k=k, n=n, activation="none")
    assert registry.resolve("matmul", m=m, k=k, n=n, activation="silu").kernel_name == name


@pytest.mark.parametrize("name", sorted(n for n, e in ENTRIES.items() if e[3] == "none"))
def test_an_identity_shape_refuses_silu(registry, name):
    m, k, n, *_ = ENTRIES[name]
    assert not registry.resolve("matmul", m=m, k=k, n=n, activation="silu")
