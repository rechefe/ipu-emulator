"""IPU cycle-functional emulator."""

from ipu_emu.activations import AaqOptions, ActivationParams
from ipu_emu.descriptors import REGFILE_SCHEMA, RegDescriptor
from ipu_emu.emulator import (
    DEFAULT_MAX_CYCLES,
    DebugAction,
    dump_xmem_to_binary,
    load_binary_to_xmem,
    load_fp32_as_fp8_to_xmem,
    load_program,
    load_program_from_binary,
    run_test,
    run_until_complete,
    run_with_debug,
)
from ipu_emu.errors import (
    DebugOnlyInstructionWarning,
    EmulatorError,
    RegisterIndexError,
    RegisterKindError,
    RegisterSizeError,
)
from ipu_emu.execute import (
    BreakResult,
    decode_instruction_word,
    emulator_only_instructions,
    execute_next_instruction,
    load_binary_instructions,
)
from ipu_emu.ipu import Ipu
from ipu_emu.ipu_math import fp8_bytes_to_fp32, fp32_to_fp8_bytes, ipu_add, ipu_mult
from ipu_emu.ipu_state import IpuState
from ipu_emu.regfile import RegFile
from ipu_emu.stats import RunStats
from ipu_emu.xmem import XMem

__all__ = [
    "DebugOnlyInstructionWarning",
    "EmulatorError",
    "RegisterIndexError",
    "RegisterKindError",
    "RegisterSizeError",
    "XMem",
    "RegDescriptor",
    "REGFILE_SCHEMA",
    "RegFile",
    "Ipu",
    "IpuState",
    "RunStats",
    "ActivationParams",
    "AaqOptions",
    "ipu_mult",
    "ipu_add",
    "decode_instruction_word",
    "load_binary_instructions",
    "emulator_only_instructions",
    "execute_next_instruction",
    "BreakResult",
    "load_program",
    "load_program_from_binary",
    "run_until_complete",
    "run_with_debug",
    "run_test",
    "load_binary_to_xmem",
    "dump_xmem_to_binary",
    "load_fp32_as_fp8_to_xmem",
    "fp32_to_fp8_bytes",
    "fp8_bytes_to_fp32",
    "DebugAction",
    "DEFAULT_MAX_CYCLES",
]
