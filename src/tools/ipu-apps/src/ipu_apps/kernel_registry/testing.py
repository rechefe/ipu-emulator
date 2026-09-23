"""Pytest helpers for kernel ``test.py`` files; imported only by tests.

Every kernel's ``test.py`` starts with::

    from ipu_apps.kernel_registry.testing import case_tests

    test_case = case_tests(__package__)

and adds only the tests that go beyond "each case runs and passes its check".
A sweep of the default case over other sizes is declared, not written::

    test_case = case_tests(__package__, sweep=[dict(rows=8, n=200), ...])
"""
import pytest

from ipu_apps.kernel_registry.cases import (
    assemble_kernel, load_cases, options_label, package_kernel, run_case,
)


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

    @pytest.mark.parametrize("case,options", params, ids=[f"{kernel}-{i}" for i in ids])
    def test_case(case, options, tmp_path_factory):
        if not assembled:
            assembled.append(assemble_kernel(kernel, tmp_path_factory.mktemp(kernel)))
        run_case(kernel, load_cases(kernel)[case], options=options, inst_path=assembled[0])

    return test_case
