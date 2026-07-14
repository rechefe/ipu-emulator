"""Host-path end-to-end parity for the fully-connected application."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ipu_apps.fully_connected import FullyConnectedApp, OUTPUT_BASE_ADDR, OUTPUT_NEURONS, SAMPLES_NUM
from ipu_ctrl_regs import CTRL_START_PULSE, MMIO_BASE, REG_CTRL_OFFSET, REG_PROG_LEN_OFFSET
from ipu_emu.emulator import dump_xmem_to_binary
from ipu_emu.host_ctrl import IpuHostController
from ipu_emu.ipu_state import IpuState, INST_MEM_SIZE

_FC_INST_BIN = Path(os.environ.get("FC_INST_BIN", ""))
_FC_DATA_DIR = Path(os.environ.get("FC_DATA_DIR", ""))


@pytest.mark.parametrize("dtype_dir,dtype_str", [
    ("int8", "int8"),
])
def test_fc_host_path_matches_direct(tmp_path: Path, dtype_dir: str, dtype_str: str) -> None:
    if not _FC_INST_BIN.exists() or not _FC_DATA_DIR.exists():
        pytest.skip("FC Bazel data not configured")

    data_dir = _FC_DATA_DIR / dtype_dir
    inputs = data_dir / f"inputs_{dtype_dir}.bin"
    weights = data_dir / f"weights_{dtype_dir}.bin"
    golden = data_dir / "out_int8_acc_int32.bin"
    if not inputs.exists() or not weights.exists():
        pytest.skip(f"Missing FC data in {data_dir}")

    direct_out = tmp_path / "direct.bin"
    app = FullyConnectedApp(
        inst_path=_FC_INST_BIN,
        inputs_path=inputs,
        weights_path=weights,
        output_path=direct_out,
        dtype=dtype_str,
    )
    direct_state, _ = app.run(max_cycles=2_000_000)

    host_out = tmp_path / "host.bin"
    ctrl = IpuHostController(IpuState(dtype=app.dtype))
    app.setup(ctrl.state)
    inst_data = _FC_INST_BIN.read_bytes()
    ctrl.load_imem_binary(inst_data)
    ctrl.engine.set_max_cycles(2_000_000)
    ctrl.write(MMIO_BASE + REG_PROG_LEN_OFFSET, INST_MEM_SIZE)
    ctrl.write(MMIO_BASE + REG_CTRL_OFFSET, CTRL_START_PULSE)
    dump_xmem_to_binary(
        ctrl.state,
        host_out,
        OUTPUT_BASE_ADDR,
        OUTPUT_NEURONS * 4,
        SAMPLES_NUM,
    )

    if golden.exists():
        assert host_out.read_bytes() == golden.read_bytes(), "host vs golden"
    assert host_out.read_bytes() == direct_out.read_bytes(), (
        f"host vs direct cycles host={ctrl.engine.cycles} direct={direct_state.stats.total_cycles}"
    )
    assert ctrl.engine.halted
