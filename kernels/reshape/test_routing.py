"""Routing of the reshape kernels (E07-04, E07-05): the ``ipu_apps`` routing
facts, on the kernel tree's registry and ``//kernels:query``.

The old router took a ``shape`` tuple; the kernels now take named parameters.
A concat query is ``concat h=.. w=.. c_a=.. c_b=..`` (the old ``shape=(H, W,
C_A, C_B)``), a fold or unfold query ``fold h=.. w=.. c=..`` (the old
``shape=(H, W, C)``), a depth-to-space query ``depth_to_space h=.. w=..
upscale_factor=..`` (the old ``shape=(r * r, H, W)``, whose channel count the
factor fixes), an identity query ``identity rows=..`` (the old ``shape=(rows,
128)``, whose width is the row's).
"""

from __future__ import annotations

import json

import pytest

from ipu_harness import query
from ipu_harness.registry import default_registry

#: Each concat entry's (h, w, c_a, c_b): the shapes of the deleted ipu_apps concat folders.
CONCATS = {
    "concat_8x8x160x160": (8, 8, 160, 160),
    "concat_16x16x128x128": (16, 16, 128, 128),
    "concat_32x32x96x96": (32, 32, 96, 96),
}


@pytest.fixture(scope="module")
def registry():
    loaded = default_registry()
    assert not loaded.invalid, loaded.invalid
    return loaded


def test_the_concats_are_three_entries_over_one_template(registry):
    assert {m.name for m in registry.kernels("concat")} == set(CONCATS)
    assert {registry.kernel(name).directory.name for name in CONCATS} == {"concat"}
    assert "concat" in registry.templates  # run and tested, not routed


@pytest.mark.parametrize("name", sorted(CONCATS))
def test_each_concat_is_routed_by_its_exact_shape(registry, name):
    h, w, c_a, c_b = CONCATS[name]
    verdict = registry.resolve("concat", h=h, w=w, c_a=c_a, c_b=c_b)
    assert verdict.kernel_name == name, verdict.reason
    assert verdict.alternatives == ()
    assert verdict.label == f"//kernels/reshape:{name}"
    assert not registry.resolve("concat", h=h, w=w, c_a=c_a, c_b=c_b + 1)


@pytest.mark.parametrize(
    ("op", "params", "label"),
    [
        ("identity", {"rows": 3}, "//kernels/reshape/identity"),
        (
            "depth_to_space",
            {"h": 2, "w": 129, "upscale_factor": 4},
            "//kernels/reshape/depth_to_space",
        ),
    ],
)
def test_the_folder_kernels_route_by_their_parameters(registry, op, params, label):
    verdict = registry.resolve(op, **params)
    assert verdict.label == label, verdict.reason
    argv = [op, *(f"{k}={v}" for k, v in params.items()), "--json"]
    assert query.main(argv, registry) == 0


def test_what_the_old_router_refused_is_refused(registry):
    assert not registry.resolve("identity", rows=0)
    verdict = registry.resolve("depth_to_space", h=2, w=8, upscale_factor=3)
    assert not verdict and "upscale_factor = 3 is not one of [1, 2, 4, 8]" in verdict.reason


def test_the_query_tool_names_a_concat(registry, capsys):
    argv = ["concat", "h=8", "w=8", "c_a=160", "c_b=160", "--json"]
    assert query.main(argv, registry) == 0
    answer = json.loads(capsys.readouterr().out)
    assert (answer["app"], answer["use"]) == (
        "concat_8x8x160x160",
        "//kernels/reshape:concat_8x8x160x160",
    )


#: The fold and unfold kernels' (h, w, c): one shape each, the MobileViT-S L3-L5.
SHAPES = {"8x8x240": (8, 8, 240), "16x16x192": (16, 16, 192), "32x32x144": (32, 32, 144)}


@pytest.mark.parametrize("op", ["unfold", "fold"])
@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_each_fold_and_unfold_is_routed_by_its_exact_shape(registry, op, shape):
    h, w, c = SHAPES[shape]
    verdict = registry.resolve(op, h=h, w=w, c=c)
    assert verdict.kernel_name == f"{op}_{shape}", verdict.reason
    assert verdict.label == f"//kernels/reshape/{op}_{shape}"
    assert not registry.resolve(op, h=h, w=w, c=c + 16)  # another channel count: no kernel


#: Each packed seam entry's (op, channels, tokens) (E07-23).
PACKED = {
    "packed_pack_192x64": ("pack", 192, 64),
    "packed_pack_240x16": ("pack", 240, 16),
    "packed_unpack_192x64": ("unpack", 192, 64),
    "packed_unpack_240x16": ("unpack", 240, 16),
}


@pytest.mark.parametrize("name", sorted(PACKED))
def test_each_packed_seam_is_routed_by_its_shape(registry, name):
    op, channels, tokens = PACKED[name]
    verdict = registry.resolve(op, channels=channels, tokens=tokens)
    assert verdict.kernel_name == name, verdict.reason
    assert registry.kernel(name).directory.name == f"packed_{op}"
