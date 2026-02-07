"""IPU cycle-functional emulator."""

from ipu_emu.xmem import XMem
from ipu_emu.descriptors import RegDescriptor, REGFILE_SCHEMA
from ipu_emu.regfile import RegFile
from ipu_emu.ipu_state import IpuState
from ipu_emu.ipu_math import DType, ipu_mult, ipu_add, fp32_to_fp8_bytes, fp8_bytes_to_fp32
from ipu_emu.execute import (
    decode_instruction_word,
    load_binary_instructions,
    execute_next_instruction,
    BreakResult,
)
from ipu_emu.emulator import (
    load_program,
    load_program_from_binary,
    run_until_complete,
    run_with_debug,
    run_test,
    load_binary_to_xmem,
    dump_xmem_to_binary,
    load_fp32_as_fp8_to_xmem,
    DebugAction,
)
from ipu_emu.debug_cli import (
    debug_prompt,
    DebugCLI,
    print_all_registers,
    state_to_json_dict,
    save_state_json,
    disassemble_current,
)

__all__ = [
    "XMem",
    "RegDescriptor",
    "REGFILE_SCHEMA",
    "RegFile",
    "IpuState",
    "DType",
    "ipu_mult",
    "ipu_add",
    "decode_instruction_word",
    "load_binary_instructions",
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
    "debug_prompt",
    "DebugCLI",
    "print_all_registers",
    "state_to_json_dict",
    "save_state_json",
    "disassemble_current",
]
