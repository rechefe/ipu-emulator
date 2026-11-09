"""The generated tests of a kernel's target (E06-13): ``test_case[<case>-<format>]``.

Each runs one case in one format through :func:`ipu_harness.runner.run_kernel`
and fails with the run's report: the case, the format, each output's worst
index and its tolerance tier. A kernel that declares its XMEM tables
(``[tables]``, :mod:`ipu_harness.tables`) has every case's run held to them
too: an error finding fails the case.
"""

from __future__ import annotations

from ipu_harness.runner import RunOptions
from ipu_harness.tables import declares_tables


def test_case(case_format, run_case, kernel):
    case, fmt = case_format
    checked = declares_tables(kernel)
    result = run_case(case, fmt, RunOptions(tables=checked))
    assert result.passed, result.describe()
    if result.tables is not None:
        errors = result.tables.errors
        assert not errors, "\n".join(
            ["[tables] disagrees with the run:", *(e.describe() for e in errors)]
        )
