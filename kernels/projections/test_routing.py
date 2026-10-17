"""Routing of the projections (E07-02): the ``ipu_apps`` router tests, on the
kernel tree's registry and ``//kernels:query``.

Each entry is found by exactly its ``(k, n_out, activation)``: a query for an
FFN1 shape with ``activation=none`` never reaches the kernel whose store fuses
SiLU, and no identity kernel accepts a SiLU query. A query that leaves
``activation`` out takes the template's default, ``none``: an identity kernel
accepts it and an FFN1 kernel refuses it, as the ``ipu_apps`` router did.
"""

from __future__ import annotations

import json

import pytest

from ipu_harness import query
from ipu_harness.registry import default_registry

#: Each kernel's (k, n_out, activation): the ipu_apps ProjectionLayout lines.
KERNELS = {
    "proj_qkv_144_p4": (144, 432, "none"),
    "proj_outproj_144_p4": (144, 144, "none"),
    "proj_ffn1_144_p4": (144, 288, "silu"),
    "proj_ffn2_144_p4": (288, 144, "none"),
    "proj_qkv_192_p4": (192, 576, "none"),
    "proj_outproj_192_p4": (192, 192, "none"),
    "proj_ffn1_192_p4": (192, 384, "silu"),
    "proj_ffn2_192_p4": (384, 192, "none"),
    "proj_qkv_240_p4": (240, 720, "none"),
    "proj_outproj_240_p4": (240, 240, "none"),
    "proj_ffn1_240_p4": (240, 480, "silu"),
    "proj_ffn2_240_p4": (480, 240, "none"),
}


@pytest.fixture(scope="module")
def registry():
    loaded = default_registry()
    assert not loaded.invalid, loaded.invalid
    return loaded


def test_the_family_is_twelve_entries_over_three_templates(registry):
    assert {m.name for m in registry.kernels("projection")} == set(KERNELS)
    template = {name: registry.kernel(name).directory.name for name in KERNELS}
    assert sorted(n for n, t in template.items() if t == "proj_p4_2tg") == [
        "proj_ffn2_144_p4",
        "proj_outproj_144_p4",
        "proj_qkv_144_p4",
    ]
    assert [n for n, t in template.items() if t == "proj_p4_2tg_silu"] == ["proj_ffn1_144_p4"]
    assert sum(t == "proj_p4" for t in template.values()) == 8
    assert set(template.values()) <= set(registry.templates)  # run and tested, not routed


@pytest.mark.parametrize("name", sorted(KERNELS))
def test_each_kernel_is_routed_by_its_exact_query(registry, name):
    k, n_out, activation = KERNELS[name]
    verdict = registry.resolve("projection", k=k, n_out=n_out, activation=activation)
    assert verdict.kernel_name == name, verdict.reason
    assert verdict.alternatives == ()
    assert verdict.label == f"//kernels/projections:{name}"


@pytest.mark.parametrize("name", sorted(KERNELS))
def test_the_wrong_activation_is_refused(registry, name):
    k, n_out, activation = KERNELS[name]
    other = "none" if activation == "silu" else "silu"
    verdict = registry.resolve("projection", k=k, n_out=n_out, activation=other)
    assert not verdict, verdict.describe()
    assert f"activation = {other!r} is not one of {[activation]}" in verdict.reason


def test_ffn1_kernels_are_exactly_the_silu_kernels(registry):
    silu = {m.name for m in registry.kernels("projection") if m.fixed["activation"] == "silu"}
    assert silu == {"proj_ffn1_144_p4", "proj_ffn1_192_p4", "proj_ffn1_240_p4"}


def test_other_stream_counts_and_unknown_activations_are_refused(registry):
    verdict = registry.resolve("projection", k=240, n_out=720, n_streams=1)
    assert not verdict
    assert "n_streams = 1 is not one of [4]" in verdict.reason
    verdict = registry.resolve("projection", k=240, n_out=720, activation="gelu")
    assert not verdict
    assert "activation = 'gelu' is not one of" in verdict.reason
    assert not registry.resolve("projection", k=200, n_out=200)  # no entry's shape


@pytest.mark.parametrize("name", sorted(KERNELS))
def test_the_query_tool_gives_the_same_answer(registry, name, capsys):
    k, n_out, activation = KERNELS[name]
    argv = ["projection", f"k={k}", f"n_out={n_out}", f"activation={activation}", "--json"]
    assert query.main(argv, registry) == 0
    answer = json.loads(capsys.readouterr().out)
    assert (answer["app"], answer["use"]) == (name, f"//kernels/projections:{name}")


def test_a_query_without_activation_means_none(registry):
    # The template's default, as the ipu_apps router read a missing activation.
    refused = registry.resolve("projection", k=192, n_out=384)
    assert not refused
    assert "activation = 'none' is not one of ['silu']" in refused.reason
    assert registry.resolve("projection", k=192, n_out=576).kernel_name == "proj_qkv_192_p4"
