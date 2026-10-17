"""Routing of the attention score kernels (E07-12), on the kernel tree's registry.

The query-major (``qk_scores``) and key-major (``attn_scores_km``) score
kernels are separate ops: the two chains are bit-different by design, and the
op is what keeps a consumer of one chain's scores from being handed the
other's. Each kernel takes exactly its layer's ``(n_tok, d)``;
``attn_scores_km`` also takes ``head``, which of the 4 heads it scores.
"""

from __future__ import annotations

import json

import pytest

from ipu_harness import query
from ipu_harness.registry import default_registry

#: Each layer's (n_tok, d): L5, L4, L3.
LAYERS = {"16x60": (16, 60), "64x48": (64, 48), "256x36": (256, 36)}


@pytest.fixture(scope="module")
def registry():
    loaded = default_registry()
    assert not loaded.invalid, loaded.invalid
    return loaded


@pytest.mark.parametrize("op", ["qk_scores", "attn_scores_km"])
def test_each_layer_routes_to_its_own_kernel_of_its_own_op(registry, op):
    assert {m.name for m in registry.kernels(op)} == {f"{op}_{layer}" for layer in LAYERS}
    for layer, (n_tok, d) in LAYERS.items():
        verdict = registry.resolve(op, n_tok=n_tok, d=d)
        assert verdict.kernel_name == f"{op}_{layer}", verdict.reason
        assert verdict.kernel.op == op and verdict.alternatives == ()


@pytest.mark.parametrize("op", ["qk_scores", "attn_scores_km"])
def test_a_shape_no_layer_has_is_refused(registry, op):
    assert not registry.resolve(op, n_tok=16, d=48)
    assert not registry.resolve(op, n_tok=0, d=60)


def test_attn_scores_km_takes_a_head_of_four(registry):
    verdict = registry.resolve("attn_scores_km", n_tok=16, d=60, head=3)
    assert verdict and verdict.params["head"] == 3
    assert registry.resolve("attn_scores_km", n_tok=16, d=60).params["head"] == 0
    refused = registry.resolve("attn_scores_km", n_tok=16, d=60, head=4)
    assert not refused and "head" in refused.reason
    # qk_scores takes no head parameter, so a query naming one is refused (the
    # old router ignored unknown names; the registry refuses them).
    assert not registry.resolve("qk_scores", n_tok=16, d=60, head=0)


def test_the_query_tool_gives_the_same_answer(registry, capsys):
    argv = ["attn_scores_km", "n_tok=64", "d=48", "head=1", "--json"]
    assert query.main(argv, registry) == 0
    answer = json.loads(capsys.readouterr().out)
    assert answer["app"] == "attn_scores_km_64x48"
    assert answer["use"].startswith("//kernels/attention/attn_scores_km_64x48")
