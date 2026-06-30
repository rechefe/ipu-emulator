"""MMIO control-register model mapped onto :class:`IpuState`."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

from ipu_ctrl_regs import (
    CTRL_CONT_PULSE,
    CTRL_HALT_PULSE,
    CTRL_IMEM_SWAP_PULSE,
    CTRL_RESET_PULSE,
    CTRL_START_PULSE,
    CTRL_STEP_PULSE,
    IMEM_BASE,
    IMEM_MAP_SIZE,
    IpuErrCode,
    MMIO_BASE,
    MMIO_SIZE,
    REG_CTRL_OFFSET,
    REG_CYCLES_OFFSET,
    REG_DSTRUCTURE_OFFSET,
    REG_DTYPE_OFFSET,
    REG_ELU_ALPHA_OFFSET,
    REG_ID_OFFSET,
    REG_MAX_CYCLES_OFFSET,
    REG_PC_OFFSET,
    REG_PROG_LEN_OFFSET,
    REG_STATUS_OFFSET,
    CR_REG_OFFSETS,
)
from ipu_emu.engine import IpuEngine, RunStatus
from ipu_emu.execute import load_binary_instructions
from ipu_emu.emulator import load_program_from_binary
from ipu_emu.ipu_config import CR_DSTRUCTURE_REG_INDEX, CR_READ_ONLY_INITIAL_VALUES
from ipu_emu.ipu_math import DType
from ipu_emu.ipu_state import IpuState, INST_MEM_SIZE

if TYPE_CHECKING:
    pass

_CTRL_REG_NAMES = {
    REG_CTRL_OFFSET: "ctrl",
    REG_STATUS_OFFSET: "status",
    REG_PC_OFFSET: "pc",
    REG_CYCLES_OFFSET: "cycles",
    REG_MAX_CYCLES_OFFSET: "max_cycles",
    REG_PROG_LEN_OFFSET: "prog_len",
    REG_DTYPE_OFFSET: "dtype",
    REG_DSTRUCTURE_OFFSET: "dstructure",
    REG_ELU_ALPHA_OFFSET: "elu_alpha",
    REG_ID_OFFSET: "id",
}


class IpuHostController:
    """Decode MMIO reads/writes against the generated register map."""

    ID_MAGIC = 0x4950
    ID_VERSION = 1

    def __init__(self, state: IpuState | None = None) -> None:
        self.state = state or IpuState()
        self.engine = IpuEngine(self.state)
        self._imem_bytes = bytearray(IMEM_MAP_SIZE)
        self._imem_used = 0

    def _mmio_offset(self, address: int) -> int:
        if MMIO_BASE <= address < MMIO_BASE + MMIO_SIZE:
            return address - MMIO_BASE
        raise KeyError(address)

    def _reject_if_running(self, code: IpuErrCode) -> bool:
        if self.engine.running:
            self.engine.set_error(code)
            return True
        return False

    def read(self, address: int) -> int:
        """Read a 32-bit word from the control MMIO window."""
        offset = self._mmio_offset(address)

        if offset == REG_ID_OFFSET:
            return (self.ID_VERSION << 16) | self.ID_MAGIC

        if offset == REG_STATUS_OFFSET:
            st = self.engine
            value = 0
            if st.running:
                value |= 1 << 0
            if st.halted:
                value |= 1 << 1
            if st.break_hit:
                value |= 1 << 2
            if st.error:
                value |= 1 << 3
            value |= (st.err_code & 0xFF) << 8
            return value

        if offset == REG_PC_OFFSET:
            return self.engine.get_pc()

        if offset == REG_CYCLES_OFFSET:
            return self.engine.cycles

        if offset == REG_MAX_CYCLES_OFFSET:
            return self.engine.ctrl.max_cycles

        if offset == REG_PROG_LEN_OFFSET:
            return self.engine.ctrl.prog_len

        if offset == REG_DTYPE_OFFSET:
            return int(self.state.dtype)

        if offset == REG_DSTRUCTURE_OFFSET:
            return self.state.regfile.get_cr(CR_DSTRUCTURE_REG_INDEX)

        if offset == REG_ELU_ALPHA_OFFSET:
            return struct.unpack("<I", struct.pack("<f", float(self.state.elu_alpha)))[0]

        if offset in CR_REG_OFFSETS.values():
            cr_idx = {v: int(k) for k, v in CR_REG_OFFSETS.items()}[offset]
            return self.state.regfile.get_cr(cr_idx)

        raise KeyError(f"Unknown MMIO offset 0x{offset:03X}")

    def write(self, address: int, value: int) -> None:
        """Write a 32-bit word to the control MMIO window."""
        offset = self._mmio_offset(address)
        value &= 0xFFFF_FFFF

        if offset == REG_CTRL_OFFSET:
            self._handle_ctrl(value)
            return

        if offset == REG_PC_OFFSET:
            if self._reject_if_running(IpuErrCode.PC_WRITE_WHILE_RUNNING):
                return
            try:
                self.engine.set_pc(value)
            except (RuntimeError, ValueError):
                self.engine.set_error(IpuErrCode.INVALID_PC)
            return

        if offset == REG_MAX_CYCLES_OFFSET:
            self.engine.set_max_cycles(value)
            return

        if offset == REG_PROG_LEN_OFFSET:
            if self._reject_if_running(IpuErrCode.PROG_LEN_WHILE_RUNNING):
                return
            self.engine.set_prog_len(value)
            return

        if offset == REG_DTYPE_OFFSET:
            self.state.dtype = DType(value & 0xFF)
            return

        if offset == REG_DSTRUCTURE_OFFSET:
            self.state.set_cr_dstructure(
                valid_elements=value & 0xFF,
                partition=(value >> 8) & 0xF,
            )
            return

        if offset == REG_ELU_ALPHA_OFFSET:
            self.state.set_activation_alphas(
                elu_alpha=struct.unpack("<f", struct.pack("<I", value))[0],
            )
            return

        if offset in CR_REG_OFFSETS.values():
            cr_idx = {v: int(k) for k, v in CR_REG_OFFSETS.items()}[offset]
            if cr_idx in CR_READ_ONLY_INITIAL_VALUES:
                self.engine.set_error(IpuErrCode.CR_READ_ONLY_WRITE)
                return
            if cr_idx == CR_DSTRUCTURE_REG_INDEX:
                self.state.set_cr_dstructure(
                    valid_elements=value & 0xFF,
                    partition=(value >> 8) & 0xF,
                )
            else:
                self.state.regfile.set_cr(cr_idx, value)
            return

        raise KeyError(f"Unknown MMIO offset 0x{offset:03X}")

    def _handle_ctrl(self, value: int) -> None:
        if value & CTRL_START_PULSE:
            self.engine.clear_error()
            self.engine.start()
        if value & CTRL_STEP_PULSE:
            self.engine.clear_error()
            self.engine.step(honor_break=True)
        if value & CTRL_HALT_PULSE:
            self.engine.halt()
        if value & CTRL_CONT_PULSE:
            self.engine.ctrl.halt_requested = False
            self.engine.start()
        if value & CTRL_RESET_PULSE:
            self.reset(preserve_imem=True)
        if value & CTRL_IMEM_SWAP_PULSE:
            pass  # single-bank model: no-op commit barrier

    def read_imem(self, address: int, size: int, _) -> bytes:
        if self.engine.running:
            self.engine.set_error(IpuErrCode.IMEM_ACCESS_WHILE_RUNNING)
            raise RuntimeError("IMEM access while running")
        if not (IMEM_BASE <= address < IMEM_BASE + IMEM_MAP_SIZE):
            raise KeyError(address)
        off = address - IMEM_BASE
        return bytes(self._imem_bytes[off : off + size])

    def write_imem(self, address: int, size: int, data: bytes) -> None:
        if self.engine.running:
            self.engine.set_error(IpuErrCode.IMEM_ACCESS_WHILE_RUNNING)
            raise RuntimeError("IMEM access while running")
        if not (IMEM_BASE <= address < IMEM_BASE + IMEM_MAP_SIZE):
            raise KeyError(address)
        off = address - IMEM_BASE
        self._imem_bytes[off : off + size] = data
        self._imem_used = max(self._imem_used, off + size)
        self._decode_imem_window(used_bytes=self._imem_used)

    def _decode_imem_window(self, used_bytes: int | None = None) -> None:
        data = bytes(self._imem_bytes)
        if used_bytes is not None:
            data = data[:used_bytes]
        instructions = load_binary_instructions(data)
        for i in range(INST_MEM_SIZE):
            self.state.inst_mem[i] = None
        for i, inst in enumerate(instructions):
            if i < INST_MEM_SIZE:
                self.state.inst_mem[i] = inst

    def load_imem_binary(self, data: bytes) -> None:
        """Load an assembled --format bin image into the IMEM map."""
        if self._reject_if_running(IpuErrCode.IMEM_ACCESS_WHILE_RUNNING):
            raise RuntimeError("IMEM load rejected while RUNNING")
        padded = bytearray(IMEM_MAP_SIZE)
        padded[: min(len(data), IMEM_MAP_SIZE)] = data[:IMEM_MAP_SIZE]
        self._imem_bytes = padded
        self._imem_used = min(len(data), IMEM_MAP_SIZE)
        self._decode_imem_window(used_bytes=self._imem_used)
        self.engine.set_prog_len(len(load_binary_instructions(data)))

    def reset(self, *, preserve_imem: bool = True) -> bytearray:
        """Soft reset engine; return preserved IMEM bytes when applicable."""
        saved_imem = bytearray(self._imem_bytes) if preserve_imem else bytearray(IMEM_MAP_SIZE)
        self.engine.reset(preserve_imem=preserve_imem)
        if preserve_imem:
            self._imem_bytes = saved_imem
        else:
            self._imem_bytes = bytearray(IMEM_MAP_SIZE)
            self._imem_used = 0
        return self._imem_bytes
