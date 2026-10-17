"""The shared references (E06-09), and a reference run end to end through
``ipu_harness.reference`` with real PyTorch."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from ipu_harness.manifest import load_manifest
from ipu_harness.reference import evaluate, load_reference
from ipu_kernels.reference import ops


def t(values: object) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64)


def test_layernorm_matches_a_hand_computed_example():
    # Rows [1, 3] and [2, 6]: mean 2 and 4, variance 1 and 4, so both
    # normalize to [-1, 1]; then gamma [2, 1] and beta [0, 10] per column.
    y = ops.layernorm(t([[1, 3], [2, 6], [0, 0]]), t([2, 1]), t([0, 10]), dim=-1)["y"]
    assert y.tolist() == [[-2.0, 11.0], [-2.0, 11.0], [0.0, 10.0]]


def test_layernorm_of_a_constant_slice_is_beta_as_the_kernel_computes():
    # No eps: a zero variance gives rsqrt = 0, not inf or NaN.
    y = ops.layernorm(t([[5, 5, 5]]), t([3, 3, 3]), t([1, 2, 3]), dim=1)["y"]
    assert y.tolist() == [[1.0, 2.0, 3.0]]


def test_layernorm_along_the_channel_axis_matches_the_kernel_layout():
    # The normalize kernels store channels on axis 0 and tokens across lanes.
    rng = np.random.default_rng(0)
    x, gamma, beta = rng.normal(size=(4, 3)), rng.normal(size=4), rng.normal(size=4)
    y = ops.layernorm(t(x), t(gamma), t(beta), dim=0)["y"].numpy()
    want = (x - x.mean(0)) / x.std(0) * gamma[:, None] + beta[:, None]
    np.testing.assert_allclose(y, want, rtol=1e-12)


def test_softmax_rows_sum_to_one():
    y = ops.softmax(t([[1, 2, 3], [-50, 0, 50], [0, 0, 0]]), dim=-1)["y"]
    np.testing.assert_allclose(y.sum(dim=-1).numpy(), 1.0, rtol=1e-12)
    assert y[2].tolist() == pytest.approx([1 / 3] * 3)


def test_silu_matches_the_closed_form():
    points = [-4.0, -1.0, 0.0, 0.5, 3.0]
    y = ops.silu(t(points))["y"].tolist()
    assert y == pytest.approx([p / (1 + math.exp(-p)) for p in points], rel=1e-12)


def test_matmul_and_unused_parameters():
    y = ops.matmul(t([[1, 2, 3], [4, 5, 6]]), t([[1, 0], [0, 1], [1, 1]]), rows=2, cols=2)["y"]
    assert y.tolist() == [[4.0, 5.0], [10.0, 11.0]]


def test_maxpool2d_pads_with_no_maximum_and_drops_a_partial_window():
    x = t([[[1, -5, 2], [-7, 3, -1], [4, -2, -9]]])  # [C=1, H=3, W=3]
    centred = ops.maxpool2d(x, kernel_size=3, stride=1, padding=1, channels=1)["y"]
    assert centred.tolist() == [[[3, 3, 3], [4, 4, 3], [4, 4, 3]]]
    assert ops.maxpool2d(x, kernel_size=2, stride=2, padding=0)["y"].tolist() == [[[3]]]


def test_channel_peak_is_the_channel_maximum_then_its_gate():
    y = ops.channel_peak(t([[1, -2, 0.5], [3, -4, 0.25]]), threshold=0.75, channels=2)["y"]
    assert y.tolist() == [[3, -2, 0.5], [2.25, 0, 0]]


def test_score_threshold_gates_and_soft_counts():
    y = ops.score_threshold(t([1, -1, 3]), threshold=1.0, temperature=2.0)["y"]
    sigmoid = [1 / (1 + math.exp(-2 * d)) for d in (0, -2, 2)]
    assert y[:3].tolist() == [0, 0, 2]
    assert y[3].item() == pytest.approx(sum(sigmoid))


def test_sample_descriptors_reproduces_a_constant_inside_the_map():
    """Bilinear weights sum to 1 where all four neighbours exist, so a constant
    map samples to itself away from the border, in both mappings; a band is
    its rows of the whole map."""
    x = torch.full((2, 3, 256), 0.5, dtype=torch.float64)
    inside = torch.full((8, 16, 256), 0.5, dtype=torch.float64)
    for stock in (False, True):
        y = ops.sample_descriptors(x, stock=stock, cell_row_start=0, cell_row_count=0)["y"]
        assert y.shape == (16, 24, 256)
        assert torch.allclose(y[4:12, 4:20], inside)
        band = ops.sample_descriptors(x, stock=stock, cell_row_start=1, cell_row_count=1)["y"]
        assert torch.equal(band, y[8:16])


def test_matmul_nt_contracts_the_last_dimension_of_both():
    a, b = t([[1, 2], [3, 4]]), t([[5, 6], [7, 8]])
    assert ops.matmul_nt(a, b, d=2)["y"].tolist() == [[17.0, 23.0], [39.0, 53.0]]
    assert ops.matmul(a, b)["y"].tolist() == [[19.0, 22.0], [43.0, 50.0]]


def test_linear_is_a_matmul_then_its_activation():
    a, b = t([[1, -2], [0.5, 3]]), t([[2, 0], [1, -1]])
    product = [[0.0, 2.0], [4.0, -3.0]]
    assert ops.linear(a, b, n_out=2)["y"].tolist() == product
    silu = ops.linear(a, b, activation="silu")["y"].tolist()
    expected = [[p / (1 + math.exp(-p)) for p in row] for row in product]
    assert silu == [pytest.approx(row, rel=1e-12) for row in expected]
    with pytest.raises(ValueError, match="'relu' is not one of"):
        ops.linear(a, b, activation="relu")
    # Output-major weights over channel-major data with leading axes.
    weights, data = (
        torch.ones(3, 2, dtype=torch.float64),
        torch.ones(4, 1, 2, 5, dtype=torch.float64),
    )
    assert ops.linear(weights, data)["y"].shape == (4, 1, 3, 5)


def test_conv2d_is_a_zero_padded_correlation_plus_bias_then_its_activation():
    # One input channel, a 3x3 kernel picking the right neighbour, padding 1.
    x = t([[[1, 2], [3, 4]]])
    w = torch.zeros(1, 1, 3, 3, dtype=torch.float64)
    w[0, 0, 1, 2] = 1.0
    shifted = [[[2.0, 0.0], [4.0, 0.0]]]
    assert ops.conv2d(x, w, padding=1, height=2)["y"].tolist() == shifted
    biased = ops.conv2d(x, w, t([-3]), padding=1)["y"].tolist()
    assert biased == [[[-1.0, -3.0], [1.0, -3.0]]]
    relu = [[[0.0, 0.0], [1.0, 0.0]]]
    assert ops.conv2d(x, w, t([-3]), padding=1, activation="relu")["y"].tolist() == relu
    assert ops.conv2d(x, w, t([-3]), padding=1, apply_relu=True)["y"].tolist() == relu
    # Stride 2 halves each side; groups = channels is depthwise.
    assert ops.conv2d(torch.ones(1, 4, 4, dtype=torch.float64), w, stride=2, padding=1)[
        "y"
    ].shape == (1, 2, 2)
    two = torch.ones(2, 2, 2, dtype=torch.float64)
    depthwise = ops.conv2d(two, w.repeat(2, 1, 1, 1), padding=1, groups=2)["y"]
    assert depthwise.tolist() == [[[1.0, 0.0], [1.0, 0.0]]] * 2
    with pytest.raises(ValueError, match="'gelu' is not one of"):
        ops.conv2d(x, w, activation="gelu")


def test_an_axis_op_needs_its_dim():
    # No default axis: a kernel that forgets [params] dim fails, not reduces wrongly.
    with pytest.raises(TypeError, match="dim"):
        ops.softmax(t([[1, 2]]), rows=1)


def test_a_kernel_names_a_shared_op_and_evaluate_runs_it(tmp_path: Path):
    kernel = tmp_path / "normalize" / "ln"
    kernel.mkdir(parents=True)
    (kernel / "ln.asm").write_text("HALT ;;\n")
    (kernel / "kernel.toml").write_text(
        '[kernel]\nop = "layernorm"\nformats = ["fp32"]\n\n'
        '[params]\nrows = { type = "int", default = 1 }\n'
        'dim = { type = "int", default = -1 }\n\n'
        '[inputs.x]\nshape = ["rows", 2]\n[inputs.gamma]\nshape = [2]\n'
        '[inputs.beta]\nshape = [2]\n\n[outputs.y]\nshape = ["rows", 2]\n\n'
        '[abi]\nx = "CR2"\ngamma = "CR3"\nbeta = "CR4"\ny = "CR5"\nrows = "CR6"\ndim = "CR7"\n\n'
        '[golden]\nop = "layernorm"\n\n[cases.default]\n'
    )
    fn = load_reference(load_manifest(kernel))
    assert fn is ops.layernorm
    x = np.array([[1.0, 3.0]], dtype=np.float32)
    out = evaluate(fn, {"x": x, "gamma": np.ones(2), "beta": np.zeros(2)}, {"rows": 1, "dim": -1})
    assert out["y"].dtype == np.float64 and out["y"].tolist() == [[-1.0, 1.0]]
    assert x.tolist() == [[1.0, 3.0]]  # inputs are copied, never changed


def test_each_op_is_its_own_module_named_for_it():
    """One module per op (E07): ``ops/<name>.py`` holds ``<name>``, re-exported."""
    import importlib

    for name in ops.__all__:
        assert getattr(ops, name).__module__ == f"ipu_kernels.reference.ops.{name}"
        assert importlib.import_module(f"ipu_kernels.reference.ops.{name}").__all__ == [name]
