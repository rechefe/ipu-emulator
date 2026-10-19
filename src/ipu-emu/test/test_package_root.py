"""The ``ipu_emu`` package root (E04-18): its curated exports, and that importing
it reaches no assembler or debugger module."""

from __future__ import annotations

import inspect
import subprocess
import sys

import ipu_emu
from ipu_emu import errors


def test_the_root_exports_the_curated_set():
    error_classes = [
        name
        for name, value in vars(errors).items()
        if inspect.isclass(value) and value.__module__ == errors.__name__
    ]
    entry_points = ["Ipu", "IpuState", "RunStats", "load_program", "run_until_complete"]
    assert sorted(ipu_emu.__all__) == sorted(entry_points + error_classes)


def test_importing_it_loads_no_assembler_or_debugger():
    # A fresh interpreter: this one has imported whatever the test runner did.
    # The target depends on ipu_emu_lib alone, so an import of ipu_as, ipu_debug
    # or ipu_harness fails as ModuleNotFoundError (check=True); one reachable
    # some other way is listed.
    code = (
        "import sys, ipu_emu\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] in {'ipu_as', 'ipu_debug', 'ipu_harness'}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]", result.stdout
