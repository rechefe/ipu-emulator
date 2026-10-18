"""What the old cases checked beyond the tolerance (convolutions/cases.py,
conv2d_case's ``check``): without ReLU a negative sum stays negative; with
ReLU a negative bias over zero weights writes exact zeros; and a seeded
program asked without a bias ignores the bias column; and the bias kernel
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
    # depthwise_conv_universal: weights -1 on an input of 5.
    y = _run(cases, run_case, "negative_outputs", has_bias=False, apply_relu=False, fpb=28)
    assert np.all(y < 0)


def test_relu_writes_exact_zeros_for_a_negative_bias(cases, run_case):
    # depthwise_conv_universal_bn_activation: zero weights, every bias below zero.
    y = _run(cases, run_case, "negative_bias", has_bias=True, apply_relu=True, fpb=25)
    assert np.all(y == 0.0)


def test_a_seeded_program_without_a_bias_ignores_the_bias_column(kernel, cases, run_case):
    # depthwise_conv_universal_relu (E07-20; the bias + ReLU twin requires the
    # bias): the case's bias column is nonzero, yet the run matches the
    # reference without a bias.
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


def test_the_bias_kernel_writes_the_bias_of_zero_weights_exactly(kernel, cases, run_case):
    # depthwise_conv_universal_bias (E07-20): zero weights, every bias below
    # zero; no ReLU clamps it, so each channel is its bias.
    fixed = dict(has_bias=True, apply_relu=False, fpb=25)
    case = next(c for c in cases if c.name == "negative_bias")
    case = at_values(case, case.name, fixed)
    plan = plan_case(kernel, case.values, ElementFormat.FP32, case=case.decl)
    b = inputs_for(case, plan).tensors["filters"][:, 0]
    y = _run(cases, run_case, "negative_bias", **fixed)
    assert np.all(b < 0)
    assert np.array_equal(y, np.broadcast_to(b[:, None, None], y.shape).astype(y.dtype))


def test_the_bias_kernel_keeps_a_negative_sum_negative(cases, run_case):
    # depthwise_conv_universal_bias: the seeded identity program over a zero bias.
    y = _run(cases, run_case, "negative_outputs", has_bias=True, apply_relu=False, fpb=25)
    assert np.all(y < 0)
