"""Application-level parity: apps produce identical results on both backends.

Complements the instruction-level corpus in ``ipu-emu-py/test/test_etiss_parity.py``
by running a real kernel end to end and comparing the whole machine state, not
just the output file.

Both datapaths are covered: ``fully_connected`` runs on the narrow INT8 / FP8
path, and the softmax kernels run in wide-vector FP32 debug mode.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_emu.etiss import is_available
from ipu_emu.xmem import XMEM_SIZE_BYTES

from ipu_apps.fully_connected import FullyConnectedApp
from ipu_apps.softmax.softmax_rows import SoftmaxRowsApp, LANES, ROW_BYTES

pytestmark = pytest.mark.skipif(
    not is_available(), reason="ETISS runner not built (set $IPU_ETISS_RUN)"
)

_FC_INST_BIN = Path(os.environ.get("FC_INST_BIN", ""))
_FC_DATA_DIR = Path(os.environ.get("FC_DATA_DIR", ""))

_COMPARED_REGISTERS = (
    "r", "r_wide_debug", "r_cyclic", "r_cyclic_wide_debug", "r_mask", "r_acc",
    "post_aaq_reg", "lr", "cr", "mult_res", "mem_bypass",
)

_SOFTMAX_ROWS_ASM = (
    Path(__file__).resolve().parents[1]
    / "src/ipu_apps/softmax/softmax_rows/softmax_rows.asm"
)


def _assert_states_match(py_state, et_state) -> None:
    assert et_state.program_counter == py_state.program_counter
    for name in _COMPARED_REGISTERS:
        assert bytes(et_state.regfile.raw(name)) == bytes(py_state.regfile.raw(name)), name
    assert bytes(et_state.xmem.read_address(0, XMEM_SIZE_BYTES)) == bytes(
        py_state.xmem.read_address(0, XMEM_SIZE_BYTES)
    )
    assert et_state.stats == py_state.stats


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
    _assert_states_match(py_state, et_state)


@pytest.mark.parametrize("rows", [1, 4, 8])
def test_softmax_rows_backends_agree(tmp_path: Path, rows: int) -> None:
    """A wide-vector FP32 kernel: 4-byte lanes and 512-byte XMEM rows."""
    inst = tmp_path / "softmax_rows.bin"
    assemble_to_bin_file(_SOFTMAX_ROWS_ASM.read_text(), str(inst))

    rng = np.random.default_rng(rows)
    logits = rng.normal(size=(rows, LANES)).astype(np.float32)
    input_path = tmp_path / "logits.bin"
    input_path.write_bytes(logits.tobytes())

    results = {}
    for backend in ("python", "etiss"):
        app = SoftmaxRowsApp(
            inst_path=inst, input_path=input_path, output_path=None, rows=rows
        )
        state, cycles = app.run(max_cycles=8_000_000, backend=backend)
        raw = bytes(state.xmem.read_address(app.output_base, rows * ROW_BYTES))
        results[backend] = (state, cycles, raw)

    py_state, py_cycles, py_raw = results["python"]
    et_state, et_cycles, et_raw = results["etiss"]

    assert et_cycles == py_cycles
    assert et_raw == py_raw
    _assert_states_match(py_state, et_state)

    # And the shared answer is actually a softmax, so neither backend is
    # agreeing on nonsense.
    expected = np.exp(logits - logits.max(axis=1, keepdims=True))
    expected /= expected.sum(axis=1, keepdims=True)
    actual = np.frombuffer(et_raw, dtype=np.float32).reshape(rows, LANES)
    assert np.abs(actual - expected).max() < 1e-6
