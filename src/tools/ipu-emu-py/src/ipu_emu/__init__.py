"""IPU cycle-functional emulator.

The package root holds the curated entry points (E04-18): the state, the
engine, the run statistics, loading and running a program, and every class of
``ipu_emu.errors``. Everything else is imported from its module
(``ipu_emu.emulator``, ``ipu_emu.execute``, ``ipu_emu.ipu_math``, ...). The
emulator imports only ``ipu_isa``, numpy and the standard library; no
assembler or debugger module.
"""

from ipu_emu.emulator import load_program, run_until_complete
from ipu_emu.errors import (
    ConfigurationError,
    DebugOnlyInstructionWarning,
    EmulatorError,
    EmulatorFault,
    RegisterIndexError,
    RegisterKindError,
    RegisterSizeError,
    UndeclaredRead,
)
from ipu_emu.ipu import Ipu
from ipu_emu.ipu_state import IpuState
from ipu_emu.stats import RunStats

__all__ = [
    "ConfigurationError",
    "DebugOnlyInstructionWarning",
    "EmulatorError",
    "EmulatorFault",
    "Ipu",
    "IpuState",
    "RegisterIndexError",
    "RegisterKindError",
    "RegisterSizeError",
    "RunStats",
    "UndeclaredRead",
    "load_program",
    "run_until_complete",
]
