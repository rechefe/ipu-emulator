"""End-to-end host-path parity with the direct-Python harness."""

from __future__ import annotations

import struct
from pathlib import Path

from ipu_ctrl_regs import (
    CTRL_RESET_PULSE,
    CTRL_START_PULSE,
    MMIO_BASE,
    REG_CTRL_OFFSET,
    REG_DTYPE_OFFSET,
    REG_PROG_LEN_OFFSET,
)
from ipu_emu.emulator import load_program_from_binary, run_until_complete
from ipu_emu.host_ctrl import IpuHostController
from ipu_emu.ipu_math import DType
from ipu_emu.ipu_state import IpuState


def _run_direct(inst_bin: Path) -> IpuState:
    state = IpuState(dtype=DType.INT8)
    load_program_from_binary(state, inst_bin)
    run_until_complete(state)
    return state


def test_host_path_halts_on_bkpt(tmp_path: Path) -> None:
    from ipu_as.lark_tree import assemble_to_bin_file

    bin_path = tmp_path / "bkpt.bin"
    assemble_to_bin_file("BKPT\n", str(bin_path))

    direct = _run_direct(bin_path)

    ctrl = IpuHostController(IpuState(dtype=DType.INT8))
    ctrl.write(MMIO_BASE + REG_DTYPE_OFFSET, int(DType.INT8))
    ctrl.load_imem_binary(bin_path.read_bytes())
    ctrl.write(MMIO_BASE + REG_PROG_LEN_OFFSET, 1)
    ctrl.write(MMIO_BASE + REG_CTRL_OFFSET, CTRL_START_PULSE)

    assert ctrl.state.is_halted == direct.is_halted
    assert ctrl.engine.cycles == direct.stats.total_cycles


def test_host_reset_preserves_imem_rerun(tmp_path: Path) -> None:
    from ipu_as.lark_tree import assemble_to_bin_file

    bin_path = tmp_path / "bkpt.bin"
    assemble_to_bin_file("BKPT\n", str(bin_path))

    ctrl = IpuHostController()
    ctrl.load_imem_binary(bin_path.read_bytes())
    ctrl.write(MMIO_BASE + REG_PROG_LEN_OFFSET, 1)
    ctrl.write(MMIO_BASE + REG_CTRL_OFFSET, CTRL_START_PULSE)
    first_cycles = ctrl.engine.cycles
    ctrl.write(MMIO_BASE + REG_CTRL_OFFSET, CTRL_RESET_PULSE)
    ctrl.write(MMIO_BASE + REG_CTRL_OFFSET, CTRL_START_PULSE)
    assert ctrl.engine.cycles == first_cycles
