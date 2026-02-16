"""Fully-connected layer test harness — Python port of fully_connected.c.

Mirrors the C test harness that:
1. Loads input activations and weights into XMEM
2. Transposes weights
3. Sets CR registers for base addresses and dtype
4. Runs the assembly program
5. Dumps output activations from XMEM

Usage::

    from ipu_apps.fully_connected import FullyConnectedApp

    app = FullyConnectedApp(
        inst_path="fc.bin",
        inputs_path="inputs.bin",
        weights_path="weights.bin",
        output_path="output.bin",
        dtype="INT8",
    )
    state, cycles = app.run()
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ipu_emu.ipu_math import DType
from ipu_emu.emulator import load_binary_to_xmem, dump_xmem_to_binary

from ipu_apps.base import IpuApp

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

# -- Constants (mirror #defines in fully_connected.c) -----------------------

SAMPLES_NUM = 10

INPUT_BASE_ADDR = 0x0000
INPUT_NEURONS = 128  # IPU__R_REG_SIZE_BYTES

WEIGHTS_BASE_ADDR = 0x20000

OUTPUT_BASE_ADDR = 0x40000
OUTPUT_NEURONS = 64

_DTYPE_MAP = {
    "INT8": DType.INT8,
    "int8": DType.INT8,
    "FP8_E4M3": DType.FP8_E4M3,
    "fp8_e4m3": DType.FP8_E4M3,
    "FP8_E5M2": DType.FP8_E5M2,
    "fp8_e5m2": DType.FP8_E5M2,
}


def parse_dtype(dtype_str: str) -> DType:
    """Parse a dtype string into a :class:`DType` enum value."""
    dt = _DTYPE_MAP.get(dtype_str)
    if dt is None:
        raise ValueError(
            f"Invalid dtype '{dtype_str}'. Supported: INT8, FP8_E4M3, FP8_E5M2"
        )
    return dt


def _load_and_transpose_weights(state: "IpuState", weights_path: str | Path) -> None:
    """Load weights from file and transpose into XMEM.

    Original: (OUTPUT_NEURONS × INPUT_NEURONS).
    Transposed: (INPUT_NEURONS × INPUT_NEURONS), zero-padded.
    """
    raw = Path(weights_path).read_bytes()
    expected = OUTPUT_NEURONS * INPUT_NEURONS
    if len(raw) < expected:
        raise ValueError(
            f"Weights file too small: {len(raw)} bytes, expected {expected}"
        )

    original: list[bytes] = []
    for j in range(OUTPUT_NEURONS):
        row_start = j * INPUT_NEURONS
        original.append(raw[row_start : row_start + INPUT_NEURONS])

    for i in range(INPUT_NEURONS):
        transposed_vector = bytearray(INPUT_NEURONS)
        for j in range(OUTPUT_NEURONS):
            transposed_vector[j] = original[j][i]
        state.xmem.write_address(WEIGHTS_BASE_ADDR + i * INPUT_NEURONS, transposed_vector)


class FullyConnectedApp(IpuApp):
    """Fully-connected layer application harness.

    Args:
        inst_path:    Path to assembled instruction binary.
        inputs_path:  Path to input activations binary.
        weights_path: Path to weights binary.
        output_path:  Optional path to write output.
        dtype:        Data type string or :class:`DType`.
    """

    def __init__(self, *, dtype: str | DType = "INT8", **kwargs) -> None:
        super().__init__(**kwargs)
        self.inputs_path = Path(self.inputs_path)
        self.weights_path = Path(self.weights_path)
        self.dtype = parse_dtype(dtype) if isinstance(dtype, str) else dtype

    def setup(self, state: "IpuState") -> None:
        state.set_cr_dtype(int(self.dtype))
        load_binary_to_xmem(
            state, self.inputs_path, INPUT_BASE_ADDR, INPUT_NEURONS, SAMPLES_NUM
        )
        _load_and_transpose_weights(state, self.weights_path)
        state.regfile.set_cr(0, INPUT_BASE_ADDR)
        state.regfile.set_cr(1, WEIGHTS_BASE_ADDR)
        state.regfile.set_cr(2, OUTPUT_BASE_ADDR)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            dump_xmem_to_binary(
                state, self.output_path,
                OUTPUT_BASE_ADDR, OUTPUT_NEURONS * 4, SAMPLES_NUM,
            )
