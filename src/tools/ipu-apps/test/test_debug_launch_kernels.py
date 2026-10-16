"""Every registered kernel enters the debugger at entry and runs to completion.

One Bazel target per kernel (``ipu_per_kernel_test``): IPU_KERNEL selects the
kernel; unset, every kernel.
"""

import os
import sys
from io import StringIO

import pytest

import ipu_debug.tui as tui
from ipu_apps.kernel_registry import kernels
from ipu_apps.kernel_registry.cases import load_cases, run_case
from ipu_apps.kernel_registry.testing import debug_only_warning, kernel_uses_debug_only_instructions
from ipu_emu.emulator import DebugAction

KERNEL = os.environ.get("IPU_KERNEL")
KERNELS = [spec.name for spec in kernels() if KERNEL in (None, spec.name)]
# A misspelled IPU_KERNEL must fail collection, not pass as zero skipped tests.
assert KERNELS, f"IPU_KERNEL={KERNEL!r} names no registered kernel"


class Terminal(StringIO):
    def isatty(self):
        return True


@pytest.fixture
def debug_terminal(monkeypatch):
    monkeypatch.setenv("IPU_DEBUG_TUI", "1")
    monkeypatch.setattr(sys, "stdin", Terminal())
    monkeypatch.setattr(sys, "stdout", Terminal())


@pytest.mark.parametrize("kernel", KERNELS)
def test_every_kernel_enters_debugger_and_completes(debug_terminal, monkeypatch, kernel):
    stops = []

    def view(state, cycle, **frontend):
        stops.append((state.program_counter, cycle))
        return DebugAction.CONTINUE

    monkeypatch.setattr(tui, "run_debug_tui", view)
    with debug_only_warning(kernel_uses_debug_only_instructions(kernel)):
        state, cycles = run_case(kernel, load_cases(kernel)["default"])
    assert state.is_halted and cycles > 0
    assert stops[0] == (0, 0)
    # Existing assembly BREAK instructions may reopen the debugger after entry.
    assert all(cycle > 0 for _, cycle in stops[1:])
