"""Fully-connected layer test harness — Python port of fully_connected.c.

Mirrors the C test harness that:
1. Loads input activations and weights into XMEM
2. Transposes weights
3. Sets CR registers for base addresses and IpuState dtype
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

import re
import struct
from pathlib import Path

from ipu_emu.ipu_math import DType
from ipu_emu.ipu_config import LR_CR_SCALAR_VALUE_MASK
from ipu_emu.emulator import load_binary_to_xmem
from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.base import IpuApp

# -- Constants (mirror #defines in fully_connected.c) -----------------------

SAMPLES_NUM = 10

INPUT_BASE_ADDR = 0x0000
INPUT_NEURONS = 128  # IPU__R_REG_SIZE_BYTES

WEIGHTS_BASE_ADDR = 0x20000

OUTPUT_BASE_ADDR = 0x40000
OUTPUT_NEURONS = 64

# XMEM .asm operands are ROW numbers (one row = 128 elements), not byte
# addresses -- see issue #179. The *_BASE_ADDR constants above stay byte
# addresses because they only drive direct state.xmem.write_address/
# read_address calls in this harness (which bypass row translation); the CR
# registers below feed the .asm's XMEM instructions instead, so they carry
# the same addresses converted to rows (divide by the 128-byte row size).
ROW_SIZE_BYTES = 128
# STR_ACC_REG always writes all 512 B of r_acc regardless of mode (issue
# #179's fixed-payload outlier), which spans 4 rows in narrow mode -- the
# stride between successive output vectors must match that, not the
# 256-byte/2-row footprint of one INT32 output vector alone, or narrow-mode
# writes would overlap.
OUTPUT_STRIDE_ROWS = 4
WIDE_ROW_SIZE_BYTES = 512


def parse_dtype(dtype_str: str) -> DType:
    """Parse a dtype string into a :class:`DType` enum value.

    Accepted formats (case-insensitive):

    - ``'int8'`` → :attr:`DType.INT8` (integer mode)
    - ``'fp8_e0'`` → :attr:`DType.INT8` (alias; treated as integer mode, not a float format)
    - ``'fp8_eX'`` where X is 1-7 -> FP8 with X exponent bits (e.g. ``'fp8_e4'``)
    - ``'fp8_e4m3'`` / ``'fp8_e5m2'`` -> aliases for ``fp8_e4`` / ``fp8_e5``
    """
    s = dtype_str.lower().strip()
    if s in ("int8", "fp8_e0"):
        return DType.INT8
    if s in ("fp8_e4m3", "e4m3"):
        return DType.E4
    if s in ("fp8_e5m2", "e5m2"):
        return DType.E5
    m = re.fullmatch(r"fp8_e([1-7])", s)
    if m:
        return DType(int(m.group(1)))
    raise ValueError(
        f"Invalid dtype '{dtype_str}'. Use 'int8', 'fp8_eX', "
        "'fp8_e4m3', or 'fp8_e5m2'."
    )


def _expand_int8_row(row: bytes | bytearray) -> bytes:
    """Expand 128 signed INT8 values into one 512-byte INT32 XMEM row."""
    signed = (value if value < 128 else value - 256 for value in row)
    return struct.pack("<128i", *signed)


def _load_inputs(state: IpuState, inputs_path: str | Path) -> None:
    """Load inputs with the row spacing required by the active mode."""
    if not state.wide_vector_debug:
        load_binary_to_xmem(
            state, inputs_path, INPUT_BASE_ADDR, INPUT_NEURONS, SAMPLES_NUM
        )
        return

    raw = Path(inputs_path).read_bytes()
    expected = SAMPLES_NUM * INPUT_NEURONS
    if len(raw) < expected:
        raise ValueError(f"Inputs file too small: {len(raw)} bytes, expected {expected}")
    for sample in range(SAMPLES_NUM):
        start = sample * INPUT_NEURONS
        state.xmem.write_address(
            INPUT_BASE_ADDR + sample * WIDE_ROW_SIZE_BYTES,
            _expand_int8_row(raw[start : start + INPUT_NEURONS]),
        )


def _load_and_transpose_weights(state: IpuState, weights_path: str | Path) -> None:
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
        if state.wide_vector_debug:
            data = _expand_int8_row(transposed_vector)
            row_size = WIDE_ROW_SIZE_BYTES
        else:
            data = transposed_vector
            row_size = ROW_SIZE_BYTES
        state.xmem.write_address(WEIGHTS_BASE_ADDR + i * row_size, data)


class FullyConnectedApp(IpuApp):
    """Fully-connected layer application harness.

    Args:
        inst_path:    Path to assembled instruction binary.
        inputs_path:  Path to input activations binary.
        weights_path: Path to weights binary.
        output_path:  Optional path to write output.
        dtype:        Data type string or :class:`DType`.
    """

    def __init__(
        self, *, dtype: str | DType = "INT8", wide_mode: bool = False, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.inputs_path = Path(self.inputs_path)
        self.weights_path = Path(self.weights_path)
        self.dtype = parse_dtype(dtype) if isinstance(dtype, str) else DType(dtype)
        self.wide_mode = wide_mode
        SPEC.guard(dtype=self.dtype, wide_mode=wide_mode)

    def setup(self, state: IpuState) -> None:
        state.dtype = self.dtype
        if state.wide_vector_debug and (
            state.wide_vector_arithmetic != WideVectorArithmetic.INT32
            or self.dtype != DType.INT8
        ):
            raise ValueError("fully-connected wide mode temporarily supports INT8 only")

        row_size = WIDE_ROW_SIZE_BYTES if state.wide_vector_debug else ROW_SIZE_BYTES
        _load_inputs(state, self.inputs_path)
        _load_and_transpose_weights(state, self.weights_path)
        # CR0=0 permanently (INPUT_BASE_ADDR=0x0000, no need to set).
        # CR1=1 permanently (can't be used for WEIGHTS_BASE_ADDR; moved to CR13).
        state.regfile.set_cr(2, OUTPUT_BASE_ADDR // row_size)
        # Stride constants for assembly `add lrX lrX crN;;` (replacing removed `incr`).
        # lr0/lr14 walk one row (one input/weight vector) per iteration.
        state.regfile.set_cr(3, 1)
        state.regfile.set_cr(4, 1)
        state.regfile.set_cr(5, 512 // row_size)
        # Constants for ``SET lr* cr*`` in ``fully_connected.asm`` (issue #82).
        # cr7 is the BLT lr0/lr1 loop bound: lr0 now increments by 1 row per
        # sample (was 128 bytes), so the bound is SAMPLES_NUM rows, not the
        # old byte count (10 * 128 = 1280).
        state.regfile.set_cr(6, 0)
        state.regfile.set_cr(7, SAMPLES_NUM)
        state.regfile.set_cr(8, 0)
        # Pre-decremented by one VLIW step because ADD runs before LDR/MULT within a cycle.
        # The 32-bit wraparound on the first ADD gives the correct starting values (0 for
        # both lr4 offset and lr5 element index on the first effective iteration).
        state.regfile.set_cr(9, (-1) & LR_CR_SCALAR_VALUE_MASK)     # lr4 cyclic offset (rows): pre-decremented
        state.regfile.set_cr(10, (-1) & LR_CR_SCALAR_VALUE_MASK)    # lr5 counter: pre-decremented
        state.regfile.set_cr(11, 127)                # BNE exit condition
        state.regfile.set_cr(12, 0)
        state.regfile.set_cr(13, WEIGHTS_BASE_ADDR // row_size)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            # STR_ACC_REG writes each sample 512 B apart (OUTPUT_STRIDE_ROWS
            # rows), but only the first OUTPUT_NEURONS*4 bytes of each are
            # real output -- dump_xmem_to_binary can't express a stride
            # different from its chunk size, so read samples individually.
            sample_stride = OUTPUT_STRIDE_ROWS * ROW_SIZE_BYTES
            sample_size = OUTPUT_NEURONS * 4
            parts = [
                bytes(state.xmem.read_address(OUTPUT_BASE_ADDR + i * sample_stride, sample_size))
                for i in range(SAMPLES_NUM)
            ]
            Path(self.output_path).write_bytes(b"".join(parts))


# The assembly has fixed dimensions; only storage/arithmetic mode varies.
from ipu_apps.kernel_registry.spec import ExecutionConfig, KernelSpec, no, yes


def _supports(**params):
    if tuple(params.get("shape", (10, 128))) != (10, 128):
        return no("fully_connected requires input shape (10, 128)")
    if tuple(params.get("weight_shape", (64, 128))) != (64, 128):
        return no("fully_connected requires weight shape (64, 128)")
    try:
        dtype = params.get("dtype", "INT8")
        dtype = parse_dtype(dtype) if isinstance(dtype, str) else DType(dtype)
    except (ValueError, TypeError) as exc:
        return no(str(exc))
    if params.get("wide_mode", False) and dtype != DType.INT8:
        return no("wide_mode requires INT8")
    return yes()


SPEC = KernelSpec(
    name="fully_connected",
    op="fully_connected",
    app_class=FullyConnectedApp,
    asm="fully_connected.asm",
    execution=lambda app: ExecutionConfig(
        mode="int32" if app.wide_mode else "native", dtype=app.dtype,
    ),
    supports=_supports,
    build=lambda **p: {"dtype": p.get("dtype", "INT8"), "wide_mode": p.get("wide_mode", False)},
    explain=lambda **_: "Fixed 10 x 128 inputs and 64 x 128 weights",
)
