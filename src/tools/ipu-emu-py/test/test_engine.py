"""Tests for the steppable IPU execution engine."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipu_emu.emulator import load_program_from_binary, run_until_complete
from ipu_emu.engine import IpuEngine, RunStatus
from ipu_emu.ipu_state import IpuState


def test_engine_run_halts_on_bkpt(tmp_path: Path) -> None:
    from ipu_as.lark_tree import assemble_to_bin_file

    bin_path = tmp_path / "halt.bin"
    assemble_to_bin_file("BKPT\n", str(bin_path))

    direct = IpuState()
    load_program_from_binary(direct, bin_path)
    direct_cycles = run_until_complete(direct)

    state = IpuState()
    load_program_from_binary(state, bin_path)
    engine = IpuEngine(state)
    engine.set_prog_len(1)
    status = engine.start()
    assert status == RunStatus.HALTED
    assert engine.cycles == direct_cycles


def test_engine_step_advances_cycles() -> None:
    state = IpuState()
    engine = IpuEngine(state)
    engine.set_prog_len(4)
    status = engine.step()
    assert status in (RunStatus.RUNNING, RunStatus.HALTED)
    assert engine.cycles == 1


def test_engine_reset_preserves_imem(tmp_path: Path) -> None:
    from ipu_as.lark_tree import assemble_to_bin_file

    bin_path = tmp_path / "halt.bin"
    assemble_to_bin_file("BKPT\n", str(bin_path))
    state = IpuState()
    load_program_from_binary(state, bin_path)
    saved = list(state.inst_mem)
    state.program_counter = 99
    engine = IpuEngine(state)
    engine.set_prog_len(1)
    engine.reset(preserve_imem=True)
    assert state.inst_mem == saved
    assert state.program_counter == 0


def test_set_pc_rejected_while_running() -> None:
    state = IpuState()
    engine = IpuEngine(state)
    engine.ctrl.running = True
    with pytest.raises(RuntimeError):
        engine.set_pc(0)
