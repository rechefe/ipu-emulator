"""The generated tests of a kernel's target (E06-13): ``test_case[<case>-<format>]``.

Each runs one case in one format through :func:`ipu_harness.runner.run_kernel`
and fails with the run's report: the case, the format, each output's worst
index and its tolerance tier.
"""

from __future__ import annotations


def test_case(case_format, run_case):
    case, fmt = case_format
    result = run_case(case, fmt)
    assert result.passed, result.describe()
