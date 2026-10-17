"""The entry of every ``ipu_kernel()`` target (E06-13).

    bazel test //kernels/<family>/<name>          # every case x format
    bazel run  //kernels/<family>/<name> -- ...   # one case: the CLI (E06-11)

Bazel sets ``BUILD_WORKING_DIRECTORY`` for ``bazel run``, test rules included,
and not for ``bazel test``; ``BAZEL_TEST`` alone cannot tell them apart (both
run test-setup.sh). Under ``bazel test`` the pytest shim of the
rules_python_pytest module runs, so its filtering, XML report and sharding
apply, over the generated tests (:mod:`ipu_harness.kernel_cases`) and, in
the default tier, the kernel folder's own ``test_*.py``. Their fixtures, and
the sharding the shim asks for, come from :mod:`ipu_harness.kernel_fixtures`.
A folder's tests may import modules beside them: the folder is on the path.

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
PLUGINS_ENV = "PYTEST_PLUGINS"


def _bazel_run() -> bool:
    return "BUILD_WORKING_DIRECTORY" in os.environ or not os.environ.get("BAZEL_TEST")


def _restore(saved: dict[str, str | None]) -> None:
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def main(argv: Sequence[str] | None = None) -> int:
    """Run the kernel's tests, or (under ``bazel run``) the CLI. Returns the exit code."""
    toml, shim, tier, *args = sys.argv[1:] if argv is None else argv
    # Not resolve(): the kernel's files stay the runfiles', not the checkout's.
    directory = Path(os.path.abspath(toml)).parent
    if _bazel_run():
        from ipu_harness import cli

        return cli.main([str(directory), *args])
    from ipu_harness import kernel_cases

    extra = [str(p) for p in sorted(directory.glob("test_*.py"))] if tier == "default" else []
    saved = {name: os.environ.get(name) for name in (KERNEL_ENV, TIER_ENV, PLUGINS_ENV)}
    os.environ[KERNEL_ENV] = str(directory)
    os.environ[TIER_ENV] = tier
    # The fixtures and sharding plugin, by environment: the shim rewrites
    # arguments that do not start with "--" under an upper-case --test_filter.
    os.environ[PLUGINS_ENV] = "ipu_harness.kernel_fixtures"
    previous_argv, previous_path = sys.argv, list(sys.path)
    sys.path.insert(0, str(directory))  # a folder's test_*.py may import its own helpers
    try:
        # Test files import by path (--import-mode=importlib), whatever the root directory.
        sys.argv = [
            shim,
            "--import-mode=importlib",
            str(Path(kernel_cases.__file__)),
            *extra,
            *args,
        ]
        runpy.run_path(shim, run_name="__main__")
    except SystemExit as exit_:
        return exit_.code if isinstance(exit_.code, int) else 1
    finally:
        sys.argv, sys.path[:] = previous_argv, previous_path
        _restore(saved)
    return 0


if __name__ == "__main__":
    sys.exit(main())
