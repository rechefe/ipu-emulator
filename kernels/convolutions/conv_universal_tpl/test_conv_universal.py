"""What the ipu_apps cases checked beyond the tolerance (convolutions/cases.py,
conv2d_case's ``check``): without ReLU a negative sum stays negative; with
ReLU a negative bias over zero weights writes exact zeros; and the bias + ReLU
twin asked without a bias ignores the bias column; and the bias kernel
(E07-20) writes a negative sum and a negative bias as they are. Each runs the
template's case at the entry's fixed values."""

import numpy as np

from ipu_harness.cases import at_values, inputs_for
from ipu_harness.params import plan_case
from ipu_isa.formats import ElementFormat


def _run(cases, run_case, name, **fixed):
    case = next(c for c in cases if c.name == name)
    case = at_values(case, case.name, fixed)
    result = run_case(case, ElementFormat.FP32)
    assert result.passed, result
    return result.outputs["y"]


def test_without_relu_every_negative_sum_stays_negative(cases, run_case):
    # conv_universal: weights -1 on an input of 5.
    y = _run(cases, run_case, "negative_outputs", has_bias=False, apply_relu=False)
    assert np.all(y < 0)


def test_relu_writes_exact_zeros_for_a_negative_bias(cases, run_case):
    # conv_universal_bn_activation: zero weights, every bias below zero.
    y = _run(cases, run_case, "negative_bias", has_bias=True, apply_relu=True)
    assert np.all(y == 0.0)


def test_relu_without_a_bias_ignores_the_bias_column(kernel, cases, run_case):
    # conv_universal_bn_activation at has_bias = false: the seeded data's bias
    # column is nonzero, yet the run matches the reference without a bias.
    case = next(c for c in cases if c.name == "relu_no_bias")
    plan = plan_case(kernel, case.values, ElementFormat.FP32, case=case.decl)
    assert np.any(inputs_for(case, plan).tensors["filters"][:, -1] != 0)
    _run(cases, run_case, "relu_no_bias", has_bias=False, apply_relu=True)


def _bias(kernel, cases, name, **fixed):
    """The bias column of case ``name`` at ``fixed``, one value per output channel."""
    case = next(c for c in cases if c.name == name)
    case = at_values(case, case.name, fixed)
    plan = plan_case(kernel, case.values, ElementFormat.FP32, case=case.decl)
    return inputs_for(case, plan).tensors["filters"][:, -1]


def test_the_bias_kernel_writes_the_bias_of_zero_weights_exactly(kernel, cases, run_case):
    # conv_universal_bias (E07-20): zero weights, every bias below zero; no
    # ReLU clamps it, so each output channel is its bias.
    fixed = dict(has_bias=True, apply_relu=False)
    b = _bias(kernel, cases, "negative_bias", **fixed)
    y = _run(cases, run_case, "negative_bias", **fixed)
    assert np.all(b < 0)
    assert np.array_equal(y, np.broadcast_to(b[:, None, None], y.shape).astype(y.dtype))


def test_the_bias_kernel_keeps_a_negative_sum_negative(cases, run_case):
    # conv_universal_bias: the seeded identity program over a zero bias.
    y = _run(cases, run_case, "negative_outputs", has_bias=True, apply_relu=False)
    assert np.all(y < 0)
