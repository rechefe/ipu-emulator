"""The entry of every ``ipu_kernel()`` target (E06-13).

    bazel test //kernels/<family>/<name>          # every case x format
    bazel run  //kernels/<family>/<name> -- ...   # one case: the CLI (E06-11)

Bazel sets ``BUILD_WORKING_DIRECTORY`` for ``bazel run``, test rules included,
and not for ``bazel test``; ``BAZEL_TEST`` alone cannot tell them apart (both
run test-setup.sh). Under ``bazel test`` the pytest shim of the
rules_python_pytest module runs, so its filtering, XML report and sharding
apply, over the generated tests (:mod:`ipu_harness.kernel_cases`) and the
kernel folder's own ``test_*.py``. Their fixtures come from
:mod:`ipu_harness.kernel_fixtures`.

Arguments (the macro passes the first three): the kernel's ``kernel.toml``,
the shim, the case tier (``default`` or ``nightly``), then the user's.
"""

from __future__ import annotations

import os
import runpy
import sys
from collections.abc import Sequence
from pathlib import Path

#: Where the generated tests find the kernel and the tier they run.
KERNEL_ENV = "IPU_KERNEL_DIR"
TIER_ENV = "IPU_KERNEL_TIER"


def _bazel_run() -> bool:
    return "BUILD_WORKING_DIRECTORY" in os.environ or not os.environ.get("BAZEL_TEST")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the kernel's tests, or (under ``bazel run``) the CLI. Returns the exit code."""
    toml, shim, tier, *args = sys.argv[1:] if argv is None else argv
    directory = Path(toml).resolve().parent
    if _bazel_run():
        from ipu_harness import cli

        return cli.main([str(directory), *args])
    os.environ[KERNEL_ENV] = str(directory)
    os.environ[TIER_ENV] = tier
    extra = [str(path) for path in sorted(directory.glob("test_*.py"))]
    previous = sys.argv
    try:
        sys.argv = [
            shim,
            "--capture=no",
            # Import test files by path, whatever the root directory (the
            # generated tests' module lives in the harness, the folder's own
            # tests in the kernel folder).
            "--import-mode=importlib",
            "-p",
            "ipu_harness.kernel_fixtures",
            "--pyargs",
            "ipu_harness.kernel_cases",
            *extra,
            *args,
        ]
        runpy.run_path(shim, run_name="__main__")
    except SystemExit as exit_:
        return int(exit_.code or 0)
    finally:
        sys.argv = previous
    return 0


if __name__ == "__main__":
    sys.exit(main())
