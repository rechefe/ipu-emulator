"""Tests for the IPU host MMIO control-register model."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from ipu_ctrl_regs import (
    CTRL_RESET_PULSE,
    CTRL_START_PULSE,
    IMEM_BASE,
    IpuErrCode,
    MMIO_BASE,
    REG_CTRL_OFFSET,
    REG_DTYPE_OFFSET,
    REG_DSTRUCTURE_OFFSET,
    REG_ELU_ALPHA_OFFSET,
    REG_PC_OFFSET,
    REG_PROG_LEN_OFFSET,
    REG_STATUS_OFFSET,
)
from ipu_emu.host_ctrl import IpuHostController
from ipu_emu.ipu_math import DType
from ipu_emu.ipu_state import IpuState


def _addr(offset: int) -> int:
    return MMIO_BASE + offset


def test_config_registers_mirror_direct_python_setup() -> None:
    ctrl = IpuHostController()
    ctrl.write(_addr(REG_DTYPE_OFFSET), int(DType.E4))
    ctrl.write(_addr(REG_DSTRUCTURE_OFFSET), (4 << 8) | 64)
    alpha_bits = struct.unpack("<I", struct.pack("<f", 0.5))[0]
    ctrl.write(_addr(REG_ELU_ALPHA_OFFSET), alpha_bits)

    direct = IpuState(dtype=DType.E4)
    direct.set_cr_dstructure(valid_elements=64, partition=4)
    direct.set_activation_alphas(elu_alpha=0.5)

    assert ctrl.state.dtype == direct.dtype
    assert ctrl.state.get_cr_dstructure() == direct.get_cr_dstructure()
    assert ctrl.state.elu_alpha == direct.elu_alpha


def test_cr0_write_sets_error() -> None:
    from ipu_ctrl_regs import REG_CR0_OFFSET

    ctrl = IpuHostController()
    before = ctrl.read(_addr(REG_STATUS_OFFSET))
    ctrl.write(_addr(REG_CR0_OFFSET), 123)
    status = ctrl.read(_addr(REG_STATUS_OFFSET))
    assert status & (1 << 3)
    assert ctrl.engine.err_code == IpuErrCode.CR_READ_ONLY_WRITE


def test_imem_load_matches_load_program_from_binary(tmp_path: Path) -> None:
    from ipu_as.lark_tree import assemble_to_bin_file
    from ipu_emu.emulator import load_program_from_binary

    bin_path = tmp_path / "prog.bin"
    assemble_to_bin_file("BKPT\n", str(bin_path))
    data = bin_path.read_bytes()

    ctrl = IpuHostController()
    ctrl.load_imem_binary(data)
    direct = IpuState()
    load_program_from_binary(direct, bin_path)
    assert ctrl.state.inst_mem[:1] == direct.inst_mem[:1]


def test_imem_access_while_running_errors() -> None:
    ctrl = IpuHostController()
    ctrl.engine.ctrl.running = True
    with pytest.raises(RuntimeError):
        ctrl.write_imem(IMEM_BASE, 4, b"\x00\x00\x00\x00")


def test_reset_preserves_imem(tmp_path: Path) -> None:
    from ipu_as.lark_tree import assemble_to_bin_file

    bin_path = tmp_path / "prog.bin"
    assemble_to_bin_file("BKPT\n", str(bin_path))
    ctrl = IpuHostController()
    ctrl.load_imem_binary(bin_path.read_bytes())
    saved = list(ctrl.state.inst_mem)
    ctrl.state.program_counter = 7
    ctrl.write(_addr(REG_CTRL_OFFSET), CTRL_RESET_PULSE)
    assert ctrl.state.inst_mem == saved
    assert ctrl.state.program_counter == 0


def test_start_runs_program(tmp_path: Path) -> None:
    from ipu_as.lark_tree import assemble_to_bin_file

    bin_path = tmp_path / "prog.bin"
    assemble_to_bin_file("BKPT\n", str(bin_path))
    ctrl = IpuHostController()
    ctrl.load_imem_binary(bin_path.read_bytes())
    ctrl.write(_addr(REG_PROG_LEN_OFFSET), 1)
    ctrl.write(_addr(REG_CTRL_OFFSET), CTRL_START_PULSE)
    status = ctrl.read(_addr(REG_STATUS_OFFSET))
    assert status & (1 << 1)  # halted
