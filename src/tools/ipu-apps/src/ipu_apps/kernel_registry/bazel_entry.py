"""Dispatch a kernel's executable test target to the app or pytest.

Bazel sets BUILD_WORKING_DIRECTORY for `bazel run`, including runs of test
rules. `bazel test` uses a hermetic test environment without that variable.
BAZEL_TEST alone cannot distinguish them: both commands use test-setup.sh.
See https://bazel.build/docs/user-manual#running-executables.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
import runpy
import signal
import sys


@contextmanager
def app_terminal():
    """Undo test-setup.sh's stdout tee for an interactive debug run only."""
    if (os.environ.get("IPU_DEBUG_TUI") != "1"
            or not sys.stdin.isatty() or sys.stdout.isatty()):
        yield
        return
    # stdin still refers to the terminal. Preserve the original stdout so
    # completion, failure and cancellation all restore the wrapper's pipe.
    sys.stdout.flush()
    output_fd = sys.stdout.fileno()
    input_fd = sys.stdin.fileno()
    foreground = os.tcgetpgrp(input_fd)
    saved = os.dup(output_fd)
    old_signal = signal.signal(signal.SIGTTOU, signal.SIG_IGN)
    try:
        # test-setup.sh starts its child in a background process group. Curses
        # needs foreground ownership for both reads and terminal mode changes.
        os.tcsetpgrp(input_fd, os.getpgrp())
        os.dup2(input_fd, output_fd)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, output_fd)
        os.close(saved)
        os.tcsetpgrp(input_fd, foreground)
        signal.signal(signal.SIGTTOU, old_signal)


def main(argv=None):
    kernel, test_file, pytest_shim, *args = sys.argv[1:] if argv is None else argv
    if "BUILD_WORKING_DIRECTORY" in os.environ or not os.environ.get("BAZEL_TEST"):
        from ipu_apps.kernel_registry.runner import main as run_app

        with app_terminal():
            return run_app(["--kernel", kernel, *args])

    # Keep the upstream shim's filtering, XML, sharding and report handling.
    # Neither pytest nor the test module is imported when running the app.
    previous = sys.argv
    try:
        sys.argv = [pytest_shim, "--capture=no", test_file, *args]
        runpy.run_path(pytest_shim, run_name="__main__")
    finally:
        sys.argv = previous


if __name__ == "__main__":
    raise SystemExit(main())
