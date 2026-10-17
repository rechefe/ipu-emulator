"""The softmax layer adapters (E07-11, ``ipu_kernels.softmax.layers``) through
``Registry.from_layer``: torch's ``dim`` convention, rank > 2 flattening, and
the softmax relatives that must never reach a softmax kernel."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from ipu_harness.packing import stored_rows
from ipu_harness.params import plan_case
from ipu_harness.placement import place
from ipu_harness.registry import UnsupportedLayer, default_registry
from ipu_isa.formats import ElementFormat
from ipu_kernels.softmax.layers import softmax_query

ROW_KERNELS = {"softmax_rows", "softmax_rows_partial", "softmax_rows_long"}


@pytest.fixture(scope="module")
def registry():
    loaded = default_registry()
    assert not loaded.invalid, loaded.invalid
    return loaded


@pytest.mark.parametrize(("dim", "same"), [(-1, 1), (-2, 0)])
def test_a_negative_dim_counts_from_the_end(registry, dim, same):
    negative = registry.from_layer(nn.Softmax(dim=dim), (32, 300))
    positive = registry.from_layer(nn.Softmax(dim=same), (32, 300))
    assert (negative.kernel_name, negative.params) == (positive.kernel_name, positive.params)


@pytest.mark.parametrize("n", [16, 128, 300])
def test_a_1d_tensor_is_one_row(registry, n):
    one_d = registry.from_layer(nn.Softmax(dim=0), (n,))
    last = registry.from_layer(nn.Softmax(dim=-1), (n,))
    row = registry.resolve("softmax", rows=1, n=n, dim=1)
    assert one_d.kernel_name == last.kernel_name == row.kernel_name
    assert one_d.params == last.params == row.params


@pytest.mark.parametrize(
    ("shape", "dim", "kernel"),
    [
        ((32, 300), 1, "softmax_rows_long"),
        ((300, 32), 0, "softmax_columns_packed"),
        ((1000, 64), 1, "softmax_rows_partial"),
        ((64, 1000), 0, "softmax_columns"),
    ],
)
def test_a_2d_tensor_routes_by_its_axis_and_plans(registry, shape, dim, kernel):
    verdict = registry.from_layer(nn.Softmax(dim=dim), shape)
    assert verdict.kernel_name == kernel, verdict.reason
    plan = plan_case(verdict.kernel, dict(verdict.params), ElementFormat.FP32)
    place(plan, {name: stored_rows(t, plan) for name, t in verdict.kernel.tensors.items()})


def test_the_axis_decides_the_kernel(registry):
    """The same (300, 16) tensor routes by which axis is reduced."""
    assert registry.from_layer(nn.Softmax(dim=1), (300, 16)).kernel_name == "softmax_rows_partial"
    assert registry.from_layer(nn.Softmax(dim=0), (300, 16)).kernel_name == "softmax_columns_packed"


@pytest.mark.parametrize(
    ("shape", "dim", "query"),
    [
        ((2, 3, 4), -1, {"rows": 6, "n": 4, "dim": 1}),  # the reduced axis last
        ((2, 2, 2, 2), 0, {"rows": 2, "width": 8, "dim": 0}),  # the reduced axis first
    ],
)
def test_a_higher_rank_tensor_is_a_batch_of_2d_problems(registry, shape, dim, query):
    assert softmax_query(shape, dim) == query
    assert registry.from_layer(nn.Softmax(dim=dim), shape)


def test_an_interior_axis_is_refused():
    """Flattening around a middle axis would transpose the others."""
    with pytest.raises(UnsupportedLayer, match="interior axis"):
        softmax_query((2, 3, 4), 1)


@pytest.mark.parametrize("dim", [2, -3, 99])
def test_an_out_of_range_dim_is_refused(registry, dim):
    with pytest.raises(UnsupportedLayer, match="out of range"):
        registry.from_layer(nn.Softmax(dim=dim), (32, 300))


def test_softmax_without_a_dim_is_refused(registry):
    with pytest.raises(UnsupportedLayer, match="explicit dim"):
        registry.from_layer(nn.Softmax(), (8, 128))


@pytest.mark.parametrize(
    ("layer", "says"),
    [(nn.LogSoftmax(dim=1), "log\\(softmax\\(x\\)\\)"), (nn.Softmin(dim=1), "softmax\\(-x\\)")],
)
def test_softmax_relatives_are_refused_by_name(registry, layer, says):
    with pytest.raises(UnsupportedLayer, match=f"{type(layer).__name__} computes {says}"):
        registry.from_layer(layer, (8, 128))


@pytest.mark.parametrize(
    ("shape", "dim"),
    [
        ((32, 300), 1),
        ((300, 32), 0),
        ((8, 128), -1),
        ((300, 16), 0),
        ((64, 1000), -2),
        ((129, 128), 1),
    ],
)
def test_the_routed_axis_is_the_one_torch_normalises(registry, shape, dim):
    """Pinned to torch itself, not to this adapter's reading of it: the axis
    whose output sums to 1 is the one the verdict's kernel reduces."""
    y = torch.softmax(torch.randn(*shape, generator=torch.Generator().manual_seed(0)), dim=dim)
    sums = y.sum(dim=dim)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    verdict = registry.from_layer(nn.Softmax(dim=dim), shape)
    assert (verdict.kernel_name in ROW_KERNELS) == (dim % 2 == 1), verdict.kernel_name
