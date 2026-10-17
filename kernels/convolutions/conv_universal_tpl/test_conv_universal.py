"""What the ipu_apps cases checked beyond the tolerance (convolutions/cases.py,
conv2d_case's ``check``): without ReLU a negative sum stays negative; with
ReLU a negative bias over zero weights writes exact zeros. Each runs the
template's case at the entry's fixed values."""

import numpy as np

from ipu_harness.cases import at_values
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
