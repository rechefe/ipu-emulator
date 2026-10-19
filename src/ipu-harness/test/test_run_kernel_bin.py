"""The run_kernel binary runs a kernel whose golden is a reference.py, which
imports torch: the binary carries it (and every ipu_kernels package, for a
family's packs and an op's shared reference; no tree kernel uses op yet, so
only the torch half is exercised here)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_the_binary_runs_a_kernel_with_a_reference():
    binary = Path(os.environ["RUN_KERNEL"])
    kernel = Path("kernels/elementwise/scale")  # the runfiles root is the working directory
    # Hermetic: the CLI resolves paths against BUILD_WORKING_DIRECTORY when set.
    env = {k: v for k, v in os.environ.items() if k != "BUILD_WORKING_DIRECTORY"}
    done = subprocess.run(
        [str(binary), str(kernel), "--case", "default"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert done.returncode == 0, done.stderr
    assert "scale/default/fp32: PASS" in done.stdout
