"""Rewrite kernels' baseline.json files from the current tree (E02-07).

    bazel run //src/tools/ipu-apps:update_kernel_baselines -- [<kernel> ...]

Runs the named kernels' ``case_tests`` (every kernel if none is named) in
this process, with the update switch set, so each case's record is written to
its kernel folder instead of compared. Run it only in a PR that changes a
kernel's results on purpose, and say why in the PR (Code conventions, "Baselines").
"""

from __future__ import annotations

import os
import sys
from importlib.resources import files

import pytest

from ipu_apps.kernel_registry import baseline
from ipu_apps.kernel_registry.registry import kernel_spec, kernels


def main(argv: list[str]) -> int:
    workspace = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if not workspace:
        print(f"Run this through Bazel: {baseline.UPDATE_COMMAND}", file=sys.stderr)
        return 2
    names = argv or sorted(spec.name for spec in kernels())
    os.environ[baseline.UPDATE_ENV] = workspace
    failed = []
    for name in names:
        test_file = files(kernel_spec(name).resource_package).joinpath("test.py")
        print(f"== {name}", flush=True)
        status = pytest.main([str(test_file), "-q", "-p", "no:cacheprovider", "-k", "test_case"])
        if status != 0:
            failed.append(name)
    if failed:
        print(f"Cases failed; their baselines may be incomplete: {failed}", file=sys.stderr)
        return 1
    print(f"Updated {len(names)} kernel baselines")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
