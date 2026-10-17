"""What the ipu_apps cases checked beyond the tolerance (convolutions/cases.py,
conv2d_case's ``check``): without ReLU a negative sum stays negative; with
ReLU a negative bias over zero weights writes exact zeros; and a seeded
program asked without a bias ignores the bias column. Each runs the
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
    # depthwise_conv_universal: weights -1 on an input of 5.
    y = _run(cases, run_case, "negative_outputs", has_bias=False, apply_relu=False, fpb=28)
    assert np.all(y < 0)


def test_relu_writes_exact_zeros_for_a_negative_bias(cases, run_case):
    # depthwise_conv_universal_bn_activation: zero weights, every bias below zero.
    y = _run(cases, run_case, "negative_bias", has_bias=True, apply_relu=True, fpb=25)
    assert np.all(y == 0.0)


def test_a_seeded_program_without_a_bias_ignores_the_bias_column(kernel, cases, run_case):
    # No entry routes has_bias = false with ReLU (the bias + ReLU twin requires
    # the bias), but the template renders it: the case's bias column is
    # nonzero, yet the run matches the reference without a bias.
    case = next(c for c in cases if c.name == "relu_no_bias")
    plan = plan_case(kernel, case.values, ElementFormat.FP32, case=case.decl)
    assert np.any(inputs_for(case, plan).tensors["filters"][:, 0] != 0)
    _run(cases, run_case, "relu_no_bias", has_bias=False, apply_relu=True, fpb=25)


def test_an_fpb_that_disagrees_with_the_seed_is_refused(kernel, cases, run_case):
    case = next(c for c in cases if c.name == "default")
    case = at_values(case, case.name, dict(has_bias=True, apply_relu=True, fpb=28))
    try:
        run_case(case, ElementFormat.FP32)
    except ValueError as err:
        assert "fpb = 28" in str(err)
    else:
        raise AssertionError("a seeded program with fpb = 28 was not refused")
