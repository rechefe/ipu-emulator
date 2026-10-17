"""The run_kernel binary runs a kernel whose golden is a reference.py, which
imports torch: the binary carries it (and the shared ipu_kernels references)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_the_binary_runs_a_kernel_with_a_reference():
    binary = Path(os.environ["RUN_KERNEL"])
    kernel = Path("kernels/elementwise/scale")  # the runfiles root is the working directory
    done = subprocess.run(
        [str(binary), str(kernel), "--case", "default"], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr
    assert "scale/default/fp32: PASS" in done.stdout
