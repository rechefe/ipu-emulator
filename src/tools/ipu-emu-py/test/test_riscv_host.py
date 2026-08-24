"""Tests for Unicorn RISC-V host integration."""

from __future__ import annotations

import pytest

from ipu_ctrl_regs import MMIO_BASE, REG_CR2_OFFSET, REG_STATUS_OFFSET
from ipu_emu.host_ctrl import IpuHostController
from ipu_emu.riscv_host import RiscvHost

unicorn = pytest.importorskip("unicorn")


def test_riscv_host_boots() -> None:
    host = RiscvHost()
    host.load_firmware(b"\x13\x00\x00\x00")  # nop-like; firmware expanded in issue 6
    host.run(count=1)


def test_mmio_read_status() -> None:
    ctrl = IpuHostController()
    status_addr = MMIO_BASE + REG_STATUS_OFFSET
    value = ctrl.read(status_addr)
    assert value & (1 << 1)  # halted by default


def test_mmio_write_callback_reaches_controller() -> None:
    ctrl = IpuHostController()
    host = RiscvHost(controller=ctrl)
    host._mmio_write(None, REG_CR2_OFFSET, 4, 42)
    assert ctrl.state.regfile.get_cr(2) == 42
