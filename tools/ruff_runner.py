"""Run the ruff executable from the locked wheel (requirements_lock.txt).

Two modes, both started through Bazel so ruff never comes from the host:

    bazel run //tools:ruff -- <ruff args>    # e.g. `check --fix`, `format`
    bazel test //tools:lint                  # every ruff_test (tools/lint.bzl)

`bazel run` runs ruff in the workspace root with the arguments given, so paths
are relative to the repository. A ruff_test runs `ruff check` and
`ruff format --check` over its files with the repository's ruff.toml.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import ruff
from ruff.__main__ import find_ruff_bin

TEST_FLAG = "--test"


def ruff_executable() -> str:
    """Path of the ruff binary shipped in the locked wheel."""
    # rules_python unpacks the wheel's scripts next to its site-packages
    # directory (<repo>/site-packages/ruff and <repo>/bin/ruff), a layout the
    # wheel's own finder does not search.
    candidate = Path(ruff.__file__).resolve().parents[2] / "bin" / "ruff"
    if candidate.is_file():
        return str(candidate)
    return find_ruff_bin()


def run_test(exe: str, config: str, files: list[str]) -> int:
    """`ruff check` and `ruff format --check` over files; nonzero if either fails."""
    if not files:
        print("no Python files to check")
        return 0
    common = ["--no-cache", "--config", config]
    status = 0
    for cmd in (["check", *common], ["format", "--check", *common]):
        print(f"$ ruff {' '.join(cmd)} <{len(files)} files>", flush=True)
        status |= subprocess.call([exe, *cmd, *files])
    if status:
        print(
            "\nFix locally with `bazel run //tools:ruff -- check --fix` and "
            "`bazel run //tools:ruff -- format`.",
            file=sys.stderr,
        )
    return status


def main(argv: list[str]) -> int:
    exe = ruff_executable()
    if argv and argv[0] == TEST_FLAG:
        return run_test(exe, config=argv[1], files=argv[2:])
    workspace = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    return subprocess.call([exe, *argv], cwd=workspace)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
