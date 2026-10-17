"""The ``ipu_emu`` package root (E04-18): its curated exports, and that importing
it reaches no assembler or debugger module."""

from __future__ import annotations

import os
import subprocess
import sys

import ipu_emu


def test_the_root_exports_the_curated_set():
    assert sorted(ipu_emu.__all__) == [
        "DebugOnlyInstructionWarning",
        "EmulatorError",
        "Ipu",
        "IpuState",
        "RegisterIndexError",
        "RegisterKindError",
        "RegisterSizeError",
        "RunStats",
        "load_program",
        "run_until_complete",
    ]
    for name in ipu_emu.__all__:
        assert getattr(ipu_emu, name) is not None, name


def test_importing_it_loads_no_assembler_or_debugger():
    # A fresh interpreter: this one has imported whatever the test runner did.
    code = (
        "import sys, ipu_emu\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] in {'ipu_as', 'ipu_debug', 'ipu_apps'}))"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env
    )
    assert result.stdout.strip() == "[]", result.stdout
