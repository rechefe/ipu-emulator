"""softmax_rows beyond its golden: a row of equal logits is uniform, and each
row's largest logit stays its largest probability."""

from __future__ import annotations

import numpy as np

from ipu_isa.formats import ElementFormat


def _run(cases, run_case, name):
    (case,) = [c for c in cases if c.name == name]
    result = run_case(case, ElementFormat.FP32)
    assert result.passed, result.describe()
    return case, result.outputs["y"]


def test_a_row_of_equal_logits_is_uniform(cases, run_case):
    _, y = _run(cases, run_case, "constant_row")
    np.testing.assert_allclose(y, 1.0 / 128, rtol=0, atol=1e-6)


def test_the_largest_logit_keeps_the_largest_probability(cases, run_case):
    case, y = _run(cases, run_case, "argmax")
    v = case.values  # the case's data, as gen.py:legacy_randn draws it
    x = np.random.RandomState(v["seed"]).randn(v["rows"], v["n"]) * v["scale"]
    x = x.astype(np.float32)
    np.testing.assert_array_equal(np.argmax(y, axis=1), np.argmax(x, axis=1))
