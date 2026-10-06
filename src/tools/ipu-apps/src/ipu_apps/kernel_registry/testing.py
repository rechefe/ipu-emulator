"""Pytest helpers for kernel ``test.py`` files; imported only by tests.

Every kernel's ``test.py`` starts with::

    from ipu_apps.kernel_registry.testing import case_tests

    test_case = case_tests(__package__)

and adds only the tests that go beyond "each case runs and passes its check".
A sweep of the default case over other sizes is declared, not written::

    test_case = case_tests(__package__, sweep=[dict(rows=8, n=200), ...])
"""
import warnings

import pytest

from ipu_apps.kernel_registry.cases import (
    assemble_kernel, load_cases, options_label, package_kernel, run_case,
)
from ipu_apps.kernel_registry.registry import kernel_spec
from ipu_apps.kernel_registry.tables import check_declared, run_traced


def case_tests(package, sweep=()):
    """A pytest test running every case of the kernel declared in ``package``.

    ``sweep`` adds runs of the default case with option overrides -- dicts,
    like a benchmark's ``CONFIGS`` -- each checked by the case exactly as the
    named cases are. The kernel is assembled once and reused. ``run_case``
    raises if the kernel does not halt or a check fails.
    """
    kernel = package_kernel(package)
    cases = load_cases(kernel)
    params = [(name, {}) for name in cases]
    ids = list(cases)
    for options in sweep:
        unknown = set(options) - set(cases["default"].defaults)
        if unknown:
            raise ValueError(f"{kernel}: sweep options {sorted(unknown)} are not options "
                             f"of its default case {sorted(cases['default'].defaults)}")
        params.append(("default", dict(options)))
        ids.append(options_label(options))
    assembled = []

    declares_tables = kernel_spec(kernel).tables is not None

    @pytest.mark.parametrize("case,options", params, ids=[f"{kernel}-{i}" for i in ids])
    def test_case(case, options, tmp_path_factory):
        if not assembled:
            assembled.append(assemble_kernel(kernel, tmp_path_factory.mktemp(kernel)))
        kernel_case = load_cases(kernel)[case]
        if not declares_tables:
            run_case(kernel, kernel_case, options=options, inst_path=assembled[0])
            return
        # A kernel that declares its tables has the declaration checked by
        # every case, so a formula is exercised at every size the cases cover.
        findings = check_declared(run_traced(kernel, kernel_case, options=options,
                                             inst_path=assembled[0]))
        for finding in findings:
            if finding.severity == "warning":
                warnings.warn(f"{kernel} SPEC.tables[{finding.table}]: {finding.message}")
        errors = [f"{f.table}: {f.message}" for f in findings if f.severity == "error"]
        if errors:
            pytest.fail(f"{kernel} SPEC.tables disagrees with the run "
                        f"(see `table_trace --kernel {kernel}`):\n  " + "\n  ".join(errors))

    return test_case
