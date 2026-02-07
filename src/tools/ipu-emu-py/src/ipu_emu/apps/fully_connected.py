"""Fully-connected layer test harness — Python port of fully_connected.c.

Mirrors the C test harness that:
1. Loads input activations and weights into XMEM
2. Transposes weights from (output_neurons × input_neurons) to
   (input_neurons × padded_input_neurons)
3. Sets CR registers for base addresses and dtype
4. Runs the assembly program
5. Dumps output activations from XMEM

Usage::

    from ipu_emu.apps.fully_connected import run_fully_connected

    state, cycles = run_fully_connected(
        inst_path="path/to/fc.bin",
        inputs_path="path/to/inputs.bin",
        weights_path="path/to/weights.bin",
        output_path="path/to/output.bin",   # optional
        dtype="INT8",
    )
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING

from ipu_emu.ipu_math import DType
from ipu_emu.emulator import (
    load_binary_to_xmem,
    dump_xmem_to_binary,
    load_fp32_as_fp8_to_xmem,
    run_test,
    DebugCallback,
)
from ipu_emu.xmem import XMEM_WIDTH_BYTES

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

# -- Constants (mirror #defines in fully_connected.c) -----------------------

SAMPLES_NUM = 10

INPUT_BASE_ADDR = 0x0000
INPUT_NEURONS = 128  # IPU__R_REG_SIZE_BYTES

WEIGHTS_BASE_ADDR = 0x20000

OUTPUT_BASE_ADDR = 0x40000
OUTPUT_NEURONS = 64

# Map string names to DType enum
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


def _load_and_transpose_weights(state: IpuState, weights_path: str | Path) -> None:
    """Load weights from file and transpose into XMEM.

    Original format: ``output_neurons`` (64) rows of ``input_neurons`` (128) bytes.
    Transposed format: ``input_neurons`` (128) rows of ``input_neurons`` (128)
    bytes, where the first ``output_neurons`` (64) bytes of each row come from
    the transposed data and the remaining bytes are zero-padded.
    """
    raw = Path(weights_path).read_bytes()
    expected = OUTPUT_NEURONS * INPUT_NEURONS
    if len(raw) < expected:
        raise ValueError(
            f"Weights file too small: {len(raw)} bytes, expected {expected}"
        )

    # Read as (OUTPUT_NEURONS × INPUT_NEURONS) matrix
    original: list[bytes] = []
    for j in range(OUTPUT_NEURONS):
        row_start = j * INPUT_NEURONS
        original.append(raw[row_start : row_start + INPUT_NEURONS])

    # Transpose + pad → (INPUT_NEURONS × INPUT_NEURONS) with first
    # OUTPUT_NEURONS bytes from transposed data
    for i in range(INPUT_NEURONS):
        transposed_vector = bytearray(INPUT_NEURONS)  # zero-padded
        for j in range(OUTPUT_NEURONS):
            transposed_vector[j] = original[j][i]
        offset = WEIGHTS_BASE_ADDR + i * INPUT_NEURONS
        state.xmem.write_address(offset, transposed_vector)


def setup_fully_connected(
    state: IpuState,
    inputs_path: str | Path,
    weights_path: str | Path,
    dtype: DType,
) -> None:
    """Setup callback: load data into XMEM and configure CR registers.

    Mirrors ``ipu_setup`` from fully_connected.c.
    """
    # Set dtype
    state.set_cr_dtype(int(dtype))

    # Load input activations (raw 8-bit data, one 128-byte row per sample)
    load_binary_to_xmem(state, inputs_path, INPUT_BASE_ADDR, INPUT_NEURONS, SAMPLES_NUM)

    # Load and transpose weights
    _load_and_transpose_weights(state, weights_path)

    # Set CR base addresses
    state.regfile.set_cr(0, INPUT_BASE_ADDR)
    state.regfile.set_cr(1, WEIGHTS_BASE_ADDR)
    state.regfile.set_cr(2, OUTPUT_BASE_ADDR)


def teardown_fully_connected(
    state: IpuState,
    output_path: str | Path,
) -> None:
    """Teardown callback: dump output activations from XMEM.

    Output is ``SAMPLES_NUM`` chunks, each ``OUTPUT_NEURONS * 4`` bytes
    (accumulator stores int32 / float32 per output neuron).
    """
    dump_xmem_to_binary(
        state,
        output_path,
        OUTPUT_BASE_ADDR,
        OUTPUT_NEURONS * 4,  # 4 bytes per word (int32 or float32)
        SAMPLES_NUM,
    )


def run_fully_connected(
    *,
    inst_path: str | Path,
    inputs_path: str | Path,
    weights_path: str | Path,
    output_path: str | Path | None = None,
    dtype: str | DType = "INT8",
    max_cycles: int = 1_000_000,
    debug_callback: DebugCallback | None = None,
) -> tuple[IpuState, int]:
    """Run the fully-connected layer end-to-end.

    This is the top-level entry point that replicates ``main()`` from
    fully_connected.c:

    1. Parse dtype.
    2. Assemble / load instruction binary.
    3. Setup: load inputs + weights, set CRs.
    4. Run the program.
    5. (Optional) dump outputs.

    Args:
        inst_path:      Path to assembled instruction binary
                        (produced by ``ipu-as assemble --format bin``).
        inputs_path:    Path to input activations binary.
        weights_path:   Path to weights binary.
        output_path:    If given, dump outputs here after execution.
        dtype:          Data type string or :class:`DType`.
        max_cycles:     Safety limit.
        debug_callback: Optional interactive debug callback.

    Returns:
        ``(final_state, cycles_executed)``
    """
    if isinstance(dtype, str):
        dtype = parse_dtype(dtype)

    def _setup(state: IpuState) -> None:
        setup_fully_connected(state, inputs_path, weights_path, dtype)

    def _teardown(state: IpuState) -> None:
        if output_path is not None:
            teardown_fully_connected(state, output_path)

    return run_test(
        inst_path=inst_path,
        setup=_setup,
        teardown=_teardown,
        max_cycles=max_cycles,
        debug_callback=debug_callback,
    )
