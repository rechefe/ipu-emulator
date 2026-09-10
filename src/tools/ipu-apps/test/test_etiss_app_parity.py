"""Application-level parity: apps produce identical results on both backends.

Complements the instruction-level corpus in ``ipu-emu-py/test/test_etiss_parity.py``
by running a real kernel end to end and comparing the whole machine state, not
just the output file.

Only narrow-datapath (INT8 / FP8) apps are covered.  The softmax kernels run in
wide-vector FP32 debug mode, which the ETISS backend deliberately does not
implement -- see ``docs/content/specs/etiss-integration.md``; the last test here
asserts that the backend says so clearly instead of producing wrong numbers.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ipu_emu.errors import EmulatorError
from ipu_emu.etiss import is_available
from ipu_emu.ipu_state import IpuState
from ipu_emu.xmem import XMEM_SIZE_BYTES

from ipu_apps.fully_connected import FullyConnectedApp

pytestmark = pytest.mark.skipif(
    not is_available(), reason="ETISS runner not built (set $IPU_ETISS_RUN)"
)

_FC_INST_BIN = Path(os.environ.get("FC_INST_BIN", ""))
_FC_DATA_DIR = Path(os.environ.get("FC_DATA_DIR", ""))

_COMPARED_REGISTERS = (
    "r", "r_cyclic", "r_mask", "r_acc", "post_aaq_reg", "lr", "cr",
    "mult_res", "mem_bypass",
)


def _run_fc(tmp_path: Path, dtype_dir: str, dtype_str: str, backend: str):
    data_dir = _FC_DATA_DIR / dtype_dir
    inputs = data_dir / f"inputs_{dtype_dir}.bin"
    weights = data_dir / f"weights_{dtype_dir}.bin"
    if not inputs.exists() or not weights.exists():
        pytest.skip(f"missing FC test data in {data_dir}")
    output = tmp_path / f"out_{backend}.bin"
    app = FullyConnectedApp(
        inst_path=_FC_INST_BIN,
        inputs_path=inputs,
        weights_path=weights,
        output_path=output,
        dtype=dtype_str,
    )
    state, cycles = app.run(max_cycles=2_000_000, backend=backend)
    return state, cycles, output.read_bytes()


@pytest.mark.parametrize("dtype_dir,dtype_str", [
    ("int8", "int8"),
    ("fp8_e4m3", "fp8_e4"),
    ("fp8_e5m2", "fp8_e5"),
])
def test_fully_connected_backends_agree(tmp_path: Path, dtype_dir: str, dtype_str: str) -> None:
    if not _FC_INST_BIN.exists():
        pytest.skip("FC_INST_BIN not set")

    py_state, py_cycles, py_out = _run_fc(tmp_path, dtype_dir, dtype_str, "python")
    et_state, et_cycles, et_out = _run_fc(tmp_path, dtype_dir, dtype_str, "etiss")

    assert et_cycles == py_cycles
    assert et_out == py_out
    assert et_state.program_counter == py_state.program_counter
    for name in _COMPARED_REGISTERS:
        assert bytes(et_state.regfile.raw(name)) == bytes(py_state.regfile.raw(name)), name
    assert bytes(et_state.xmem.read_address(0, XMEM_SIZE_BYTES)) == bytes(
        py_state.xmem.read_address(0, XMEM_SIZE_BYTES)
    )
    assert et_state.stats == py_state.stats


def test_wide_vector_mode_is_refused() -> None:
    """Wide-vector debug mode must fail loudly rather than silently diverge."""
    from ipu_emu.etiss import EtissRunner

    state = IpuState(wide_vector_debug=True)
    with pytest.raises(EmulatorError, match="wide-vector"):
        EtissRunner().run(state)
