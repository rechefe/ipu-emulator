"""The runner without torch (E06-11): a case with expected files needs no reference.

This target has no torch dependency, so an import of torch would fail; the
test also checks torch is not loaded. A spy shows each run plans its case
exactly once (E06-03).
"""

from __future__ import annotations

import sys

from scale_kernel import write_scale

import ipu_harness
from ipu_harness import params
from ipu_harness.cases import expand
from ipu_harness.runner import run_kernel


def test_expected_files_need_no_torch(tmp_path):
    manifest = write_scale(tmp_path, with_reference=False)
    for case in expand(manifest):
        for fmt in case.formats:
            result = run_kernel(manifest, case, fmt)
            assert result.passed, result.describe()
    assert "torch" not in sys.modules


def test_each_run_plans_its_case_once(tmp_path, monkeypatch):
    manifest = write_scale(tmp_path, with_reference=False)
    calls = []
    real = params.plan_case

    def spy(*args, **kwargs):
        calls.append(args[2])
        return real(*args, **kwargs)

    # Every module of the package that holds plan_case under that name.
    for module in list(sys.modules.values()):
        name = getattr(module, "__name__", "")
        if name.startswith(ipu_harness.__name__) and getattr(module, "plan_case", None) is real:
            monkeypatch.setattr(module, "plan_case", spy)
    case = next(c for c in expand(manifest) if c.name == "default")
    run_kernel(manifest, case, case.formats[0])
    assert calls == [case.formats[0]]
