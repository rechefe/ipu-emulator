"""Pytest helpers for kernel ``test.py`` files; imported only by tests.

Every kernel's ``test.py`` starts with::

    from ipu_apps.kernel_registry.testing import case_tests

    test_case = case_tests(__package__)

and adds only the tests that go beyond "each case runs and passes its check".
A sweep of the default case over other sizes is declared, not written::

    test_case = case_tests(__package__, sweep=[dict(rows=8, n=200), ...])
"""
import re
from pathlib import Path

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


# -- documented routing tables ------------------------------------------------
#
# Each ``docs/content/kernels/*.md`` page states, as a markdown table, which
# kernel handles which shape. That table is checked-in text, so it is a second
# copy of the routing rules and the copy that goes stale -- the exact failure
# the registry exists to remove. ``softmax.md`` states ranges and its family
# test re-probes the boundaries; the other pages state one exact shape per
# kernel, so a family test only has to resolve each documented row and compare.


def doc_page(name):
    """``docs/content/kernels/<name>`` as a path, or None if it is not here.

    Runfiles and a plain checkout both put the docs at the same place relative
    to the repo root, so walk up until the page appears.
    """
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "docs/content/kernels" / name
        if candidate.exists():
            return candidate
    return None


def doc_table(page, heading):
    """Rows of the markdown table under ``## <heading>`` in ``page``.

    Returns a list of dicts keyed by the table's own header cells, values as
    written (``| `matmul_128x128` | 128 |`` -> ``{"Kernel": "matmul_128x128",
    "M": "128"}``); backticks and bold markers are stripped so the keys and
    values match what the registry calls things. Stops at the next heading, so
    one page can carry a table per operation.
    """
    section = re.search(
        rf"^##\s+{re.escape(heading)}\s*$(.*?)(?=^##\s|\Z)",
        page.read_text(encoding="utf-8"), re.S | re.M,
    )
    if section is None:
        return []
    rows, header = [], None
    for line in section[1].splitlines():
        line = line.strip()
        if not line.startswith("|"):
            if header is not None:
                break       # table ended; a later table needs its own heading
            continue
        cells = [c.strip().strip("`").strip("*").strip() for c in line.strip("|").split("|")]
        if header is None:
            header = cells
        elif not all(set(c) <= set("-: ") for c in cells):
            rows.append(dict(zip(header, cells)))
    return rows
