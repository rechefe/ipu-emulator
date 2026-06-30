"""Unicorn RISC-V host integration with IPU MMIO bridge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ipu_ctrl_regs import IMEM_BASE, IMEM_MAP_SIZE, MMIO_BASE, MMIO_SIZE, RAM_BASE, RAM_SIZE
from ipu_emu.host_ctrl import IpuHostController

try:
    from unicorn import Uc, UC_ARCH_RISCV, UC_HOOK_MEM_READ, UC_HOOK_MEM_WRITE, UC_MODE_RISCV32
    from unicorn.riscv_const import UC_RISCV_REG_PC
except ImportError:  # pragma: no cover - optional until unicorn is installed
    Uc = None  # type: ignore[misc, assignment]
    UC_ARCH_RISCV = UC_MODE_RISCV32 = UC_HOOK_MEM_READ = UC_HOOK_MEM_WRITE = 0
    UC_RISCV_REG_PC = 0


@dataclass
class RiscvHostConfig:
    ram_base: int = RAM_BASE
    ram_size: int = RAM_SIZE
    mmio_base: int = MMIO_BASE
    mmio_size: int = MMIO_SIZE
    imem_base: int = IMEM_BASE
    imem_size: int = IMEM_MAP_SIZE


class RiscvHost:
    """Emulated RISC-V core driving the IPU through MMIO."""

    def __init__(
        self,
        controller: IpuHostController | None = None,
        config: RiscvHostConfig | None = None,
    ) -> None:
        if Uc is None:
            raise ImportError("unicorn is required for RiscvHost")
        self.config = config or RiscvHostConfig()
        self.controller = controller or IpuHostController()
        self._uc = Uc(UC_ARCH_RISCV, UC_MODE_RISCV32)
        self._setup_maps()

    def _setup_maps(self) -> None:
        cfg = self.config
        self._uc.mem_map(cfg.ram_base, cfg.ram_size)

        def mmio_read(uc: Any, offset: int, size: int, _unused: int) -> int:
            return self._mmio_read(uc, offset, size)

        def mmio_write(uc: Any, offset: int, size: int, value: int, _unused: int) -> None:
            self._mmio_write(uc, offset, size, value)

        def imem_read(uc: Any, offset: int, size: int, _unused: int) -> int:
            return self._imem_read(uc, offset, size)

        def imem_write(uc: Any, offset: int, size: int, value: int, _unused: int) -> None:
            self._imem_write(uc, offset, size, value)

        self._uc.mmio_map(cfg.mmio_base, cfg.mmio_size, mmio_read, None, mmio_write, None)
        self._uc.mmio_map(cfg.imem_base, cfg.imem_size, imem_read, None, imem_write, None)

    def _mmio_read(self, _uc: Any, offset: int, size: int) -> int:
        addr = self.config.mmio_base + offset
        value = 0
        for i in range(size):
            shift = i * 8
            if size == 1:
                word = self.controller.read(addr & ~3)
                value = (word >> ((addr & 3) * 8)) & 0xFF
                break
            if i == 0:
                word = self.controller.read(addr)
            value |= ((word >> shift) & 0xFF) << shift
        return value

    def _mmio_write(self, _uc: Any, offset: int, size: int, value: int) -> None:
        addr = self.config.mmio_base + offset
        if size == 4 and (addr % 4) == 0:
            self.controller.write(addr, value & 0xFFFF_FFFF)
            return
        current = self.controller.read(addr & ~3)
        for i in range(size):
            byte_val = (value >> (i * 8)) & 0xFF
            byte_addr = addr + i
            shift = (byte_addr & 3) * 8
            mask = ~(0xFF << shift) & 0xFFFF_FFFF
            current = (current & mask) | (byte_val << shift)
        self.controller.write(addr & ~3, current)

    def _imem_read(self, _uc: Any, offset: int, size: int) -> int:
        data = self.controller.read_imem(self.config.imem_base + offset, size, None)
        value = 0
        for i, b in enumerate(data):
            value |= b << (i * 8)
        return value

    def _imem_write(self, _uc: Any, offset: int, size: int, value: int) -> None:
        data = bytes((value >> (i * 8)) & 0xFF for i in range(size))
        self.controller.write_imem(self.config.imem_base + offset, size, data)

    def load_firmware(self, image: bytes, entry: int | None = None) -> None:
        base = self.config.ram_base
        self._uc.mem_write(base, image)
        self._uc.reg_write(UC_RISCV_REG_PC, entry if entry is not None else base)

    def run(self, until: int | None = None, count: int | None = None) -> None:
        base = self.config.ram_base
        end = until if until is not None else base + self.config.ram_size
        if count is not None:
            self._uc.emu_start(base, end, count=count)
        else:
            self._uc.emu_start(base, end)

    @property
    def uc(self) -> Any:
        return self._uc

    @property
    def ipu(self) -> IpuHostController:
        return self.controller
