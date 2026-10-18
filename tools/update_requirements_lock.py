"""Refresh requirements_lock.txt with `uv pip compile`.

Run through Bazel, which supplies uv from the lock itself:

    bazel run //tools:update_requirements_lock [-- <extra uv pip compile args>]

Without extra arguments, uv keeps every version already in the lock, so an
unchanged tree produces a byte-identical file. Pass `--upgrade-package NAME` to
move one package, or `--upgrade` to move all of them.
"""

from __future__ import annotations

import os
import subprocess
import sys

from uv import find_uv_bin

# Inputs, relative to the workspace root. The order is part of the lock header.
INPUTS = (
    "src/tools/ipu-as/pyproject.toml",
    "src/tools/ipu-emu/pyproject.toml",
    "src/tools/ipu-isa/pyproject.toml",
    "src/tools/ipu-debug/pyproject.toml",
    "src/tools/ipu-gen/pyproject.toml",
    "src/tools/ipu-kernels/pyproject.toml",
    "src/tools/ipu-harness/pyproject.toml",
    "tools/requirements-tools.in",
)

# The repository's own packages: Bazel wires them as py_library targets.
LOCAL_PACKAGES = (
    "ipu-as",
    "ipu-emu",
    "ipu-isa",
    "ipu-debug",
    "ipu-gen",
    "ipu-kernels",
    "ipu-harness",
)

LOCK = "requirements_lock.txt"

# Resolve for exactly the hermetic interpreter (MODULE.bazel pins 3.10) on a
# fixed platform, so the lock does not depend on the host that refreshes it.
# Not --universal: a universal lock forks numpy, pyslang and others into one
# line per Python version, and rules_python 0.40's pip.parse keeps only the
# last line of a duplicated name (it installed numpy for Python >= 3.12).
# --generate-hashes still lists every wheel of each pinned version, so the
# same lock installs on macOS as long as no dependency is marked macOS-only.
PLATFORM = "x86_64-unknown-linux-gnu"


def compile_command(uv: str, extra: list[str]) -> list[str]:
    cmd = [uv, "pip", "compile", *INPUTS]
    cmd += ["--extra", "dev", "--extra", "docs"]
    cmd += ["--python-version", "3.10", "--python-platform", PLATFORM]
    cmd += ["--generate-hashes"]
    for name in LOCAL_PACKAGES:
        cmd += ["--no-emit-package", name]
    cmd += ["-o", LOCK]
    return cmd + extra


def main(argv: list[str]) -> int:
    workspace = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if not workspace:
        print(
            "error: run this with `bazel run //tools:update_requirements_lock`",
            file=sys.stderr,
        )
        return 2
    env = dict(os.environ)
    # Bazel's runfiles Python path must not leak into uv's own interpreter
    # discovery; --python-version alone selects the target.
    env.pop("PYTHONPATH", None)
    return subprocess.call(compile_command(find_uv_bin(), argv), cwd=workspace, env=env)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
