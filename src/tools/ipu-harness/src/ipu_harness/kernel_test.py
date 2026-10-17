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

Arguments (the macro passes the first three): the kernel, the shim, the case
tier (``default`` or ``nightly``), then the user's. The kernel is its
``kernel.toml``, or ``<registry.toml>::<name>`` for a family entry (E06-17,
``ipu_kernel_family()``). An entry runs the generated tests only: its
template's own ``test_*.py`` run under the template's target.
"""

from __future__ import annotations

import os
import runpy
import sys
from collections.abc import Sequence
from pathlib import Path

#: Where the generated tests find the kernel (a folder, or ``<registry.toml>::<name>``)
#: and the tier they run.
KERNEL_ENV = "IPU_KERNEL"
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


#: Between a ``registry.toml`` and an entry's name (``ipu_harness.family``,
#: which this light module does not import).
_ENTRY_SEPARATOR = "::"


def kernel_ref(arg: str) -> tuple[str, Path | None]:
    """The kernel an argument names, absolute, and the folder whose tests run.

    Not resolve(): the kernel's files stay the runfiles', not the checkout's.
    A folder kernel is its folder; an entry is ``<registry.toml>::<name>`` and
    runs no folder's tests.
    """
    path, sep, name = arg.partition(_ENTRY_SEPARATOR)
    if sep:
        return f"{os.path.abspath(path)}{sep}{name}", None
    directory = Path(os.path.abspath(path)).parent
    return str(directory), directory


def main(argv: Sequence[str] | None = None) -> int:
    """Run the kernel's tests, or (under ``bazel run``) the CLI. Returns the exit code."""
    kernel, shim, tier, *args = sys.argv[1:] if argv is None else argv
    ref, directory = kernel_ref(kernel)
    if _bazel_run():
        from ipu_harness import cli

        return cli.main([ref, *args])
    from ipu_harness import kernel_cases

    own = directory if tier == "default" else None
    extra = [str(p) for p in sorted(own.glob("test_*.py"))] if own is not None else []
    saved = {name: os.environ.get(name) for name in (KERNEL_ENV, TIER_ENV, PLUGINS_ENV)}
    os.environ[KERNEL_ENV] = ref
    os.environ[TIER_ENV] = tier
    # The fixtures and sharding plugin, by environment: the shim rewrites
    # arguments that do not start with "--" under an upper-case --test_filter.
    os.environ[PLUGINS_ENV] = "ipu_harness.kernel_fixtures"
    previous_argv, previous_path = sys.argv, list(sys.path)
    if directory is not None:
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
