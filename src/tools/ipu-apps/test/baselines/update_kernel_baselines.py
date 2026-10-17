"""Rewrite kernels' baseline.json files from the current tree (E02-07).

    bazel run //src/tools/ipu-apps:update_kernel_baselines -- [<kernel> ...]

Runs the named kernels' ``case_tests`` (every kernel if none is named) in
this process with the ``UpdateBaselines`` plugin, so each case's record is
written to its kernel folder instead of compared. Run it only in a PR that changes a
kernel's results on purpose, and say why in the PR (Code conventions, "Baselines").
"""

from __future__ import annotations

import sys
from importlib.resources import files

import baseline_support
import pytest

from ipu_apps.kernel_registry import baseline
from ipu_apps.kernel_registry.registry import kernel_spec, kernels
from ipu_apps.kernel_registry.testing import UpdateBaselines


def main(argv: list[str]) -> int:
    root = baseline_support.workspace(baseline.UPDATE_COMMAND)
    if root is None:
        return 2
    names = argv or sorted(spec.name for spec in kernels())
    failed = []
    for name in names:
        test_file = files(kernel_spec(name).resource_package).joinpath("test.py")
        print(f"== {name}", flush=True)
        args = [str(test_file), "-q", "-p", "no:cacheprovider", "-k", "test_case"]
        status = pytest.main(args, plugins=[UpdateBaselines(root)])
        if status != 0:
            failed.append(name)
    if failed:
        print(f"Cases failed; their baselines may be incomplete: {failed}", file=sys.stderr)
        return 1
    print(f"Updated {len(names)} kernel baselines")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
