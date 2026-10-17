"""Routing of the attention kernels (E07-12, E07-13), on the kernel tree's registry.

Two chains, bit-different by design: query-major (``qk_scores`` ->
``attn_v``, through AGG) and key-major (``attn_scores_km`` ->
``attn_v_bcast``, through ACC.ADD). Each stage of each chain is its own op,
so the registry never hands one chain's scores to the other chain's consumer:
both attn@V kernels of a layer take the same ``(n_tok, d)``, and only the op
tells them apart. Each kernel takes exactly its layer's ``(n_tok, d)``;
``attn_scores_km`` also takes ``head``, which of the 4 heads it scores, and
``attn_v_bcast`` may leave ``n_tok`` out (it is named by ``d``).
"""

from __future__ import annotations

import json

import pytest

from ipu_harness import query
from ipu_harness.registry import default_registry

#: Each layer's (n_tok, d): L5, L4, L3.
LAYERS = {"16x60": (16, 60), "64x48": (64, 48), "256x36": (256, 36)}

#: Each chain's ops, producer first.
CHAINS = {"query-major": ("qk_scores", "attn_v"), "key-major": ("attn_scores_km", "attn_v_bcast")}
OPS = [op for ops in CHAINS.values() for op in ops]


def kernel_name(op: str, layer: str) -> str:
    """``op``'s kernel at ``layer``: ``attn_v_bcast`` is named by ``d`` alone."""
    return f"{op}_{layer.split('x')[1]}" if op == "attn_v_bcast" else f"{op}_{layer}"


@pytest.fixture(scope="module")
def registry():
    loaded = default_registry()
    assert not loaded.invalid, loaded.invalid
    return loaded


def test_every_kernel_is_one_op_of_one_chain(registry):
    family = [m for m in registry.kernels() if m.directory.parent.name == "attention"]
    assert len(family) == len(OPS) * len(LAYERS)
    assert all(m.op in OPS for m in family), {m.name: m.op for m in family}
    names = [m.name for op in OPS for m in registry.kernels(op)]
    assert sorted(names) == sorted(m.name for m in family)


@pytest.mark.parametrize("op", OPS)
def test_each_layer_routes_to_its_own_kernel_of_its_own_op(registry, op):
    """Whatever the layer, an op resolves only to its own kernel: never the
    other chain's, which takes the same ``(n_tok, d)``."""
    assert {m.name for m in registry.kernels(op)} == {kernel_name(op, la) for la in LAYERS}
    for layer, (n_tok, d) in LAYERS.items():
        verdict = registry.resolve(op, n_tok=n_tok, d=d)
        assert verdict.kernel_name == kernel_name(op, layer), verdict.reason
        assert verdict.kernel.op == op and verdict.alternatives == ()


def test_attn_v_bcast_takes_n_tok_only_as_its_layers(registry):
    assert registry.resolve("attn_v_bcast", d=36).kernel_name == "attn_v_bcast_36"
    refused = registry.resolve("attn_v_bcast", n_tok=16, d=36)
    assert not refused and "n_tok" in refused.reason and "256" in refused.reason


@pytest.mark.parametrize("op", OPS)
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
