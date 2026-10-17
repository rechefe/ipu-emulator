"""What the ipu_apps cases checked beyond the tolerance (convolutions/cases.py,
conv2d_case's ``check``): with ReLU a negative bias over zero weights writes
exact zeros; and the bias + ReLU twin asked without a bias ignores ``b``. Each
runs the template's case at the entry's fixed values."""

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


def test_relu_writes_exact_zeros_for_a_negative_bias(cases, run_case):
    # pointwise_conv_unified_bn_activation: zero weights, every bias below zero.
    y = _run(cases, run_case, "negative_bias", has_bias=True, apply_relu=True)
    assert np.all(y == 0.0)


def test_relu_without_a_bias_ignores_the_bias_tensor(kernel, cases, run_case):
    # pointwise_conv_unified_bn_activation at has_bias = false: the case's b is
    # nonzero, yet the run matches the reference without a bias.
    case = next(c for c in cases if c.name == "relu_no_bias")
    plan = plan_case(kernel, case.values, ElementFormat.FP32, case=case.decl)
    assert np.any(inputs_for(case, plan).tensors["b"] != 0)
    _run(cases, run_case, "relu_no_bias", has_bias=False, apply_relu=True)
