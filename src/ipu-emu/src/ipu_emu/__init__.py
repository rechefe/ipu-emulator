"""The IPU cycle-functional emulator: machine state, the cycle module, per-slot
handlers, prototyping mode and the stable reference-model API (``ipu_emu.reference``).

The package root holds the curated entry points (E04-18): the state, the
engine, the run statistics, loading and running a program, and every class of
``ipu_emu.errors``. Everything else is imported from its module
(``ipu_emu.emulator``, ``ipu_emu.geometry``, ...). The
emulator imports only ``ipu_isa``, numpy and the standard library; no
assembler or debugger module.
"""

from ipu_emu.emulator import load_program, run_until_complete
from ipu_emu.errors import (
    ConfigurationError,
    CrContentError,
    DebugOnlyInstructionWarning,
    EmulatorError,
    EmulatorFault,
    RegisterIndexError,
    RegisterKindError,
    RegisterSizeError,
    UndeclaredRead,
    XMemBoundsError,
)
from ipu_emu.ipu import Ipu
from ipu_emu.ipu_state import IpuState
from ipu_emu.stats import RunStats

__all__ = [
    "ConfigurationError",
    "CrContentError",
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
    "XMemBoundsError",
    "load_program",
    "run_until_complete",
]
