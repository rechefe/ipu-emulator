"""IPU cycle-functional emulator.

The package root holds the curated entry points (E04-18): the state, the
engine, the run statistics, loading and running a program, and the error
classes. Everything else is imported from its module (``ipu_emu.emulator``,
``ipu_emu.execute``, ``ipu_emu.ipu_math``, ...). The emulator imports only
``ipu_isa``, numpy and the standard library; no assembler or debugger module.
"""

from ipu_emu.emulator import load_program, run_until_complete
from ipu_emu.errors import (
    DebugOnlyInstructionWarning,
    EmulatorError,
    RegisterIndexError,
    RegisterKindError,
    RegisterSizeError,
)
from ipu_emu.ipu import Ipu
from ipu_emu.ipu_state import IpuState
from ipu_emu.stats import RunStats

__all__ = [
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
