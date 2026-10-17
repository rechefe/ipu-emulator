"""Fully-connected layer test harness — Python port of fully_connected.c.

Mirrors the C test harness that:
1. Loads input activations and weights into XMEM
2. Transposes weights
3. Sets CR registers for base addresses and checks the state runs in its element format
4. Runs the assembly program
5. Dumps output activations from XMEM

Usage::

    from ipu_apps.kernels.linear.fully_connected.app import FullyConnectedApp

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

from ipu_apps.kernel_registry.base import IpuApp
from ipu_apps.kernel_registry.spec import ExecutionConfig, folder_spec, no, yes
from ipu_emu.emulator import load_binary_to_xmem
from ipu_emu.errors import ConfigurationError
from ipu_emu.ipu_state import IpuState
from ipu_isa.constants import LR_CR_SCALAR_VALUE_MASK
from ipu_isa.formats import ElementFormat

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
# ST.ACC always writes all 512 B of r_acc regardless of mode (issue
# #179's fixed-payload outlier), which spans 4 rows in narrow mode -- the
# stride between successive output vectors must match that, not the
# 256-byte/2-row footprint of one INT32 output vector alone, or narrow-mode
# writes would overlap.
OUTPUT_STRIDE_ROWS = 4
WIDE_ROW_SIZE_BYTES = 512


#: The 8-bit formats by exponent bits: INT8, E1..E7.
_EIGHT_BIT = (ElementFormat.INT8,) + tuple(ElementFormat(f"e{x}") for x in range(1, 8))


def parse_dtype(dtype: str | int | ElementFormat) -> ElementFormat:
    """Parse a dtype parameter into the 8-bit :class:`ElementFormat` it names.

    Accepted values (strings case-insensitive):

    - an :class:`ElementFormat` of 8 bits (``INT8``, ``E1``..``E7``), or its value
      (``'int8'``, ``'e4'``)
    - ``'fp8_e0'`` → ``INT8`` (alias; treated as integer mode, not a float format)
    - ``'fp8_eX'`` where X is 1-7 -> FP8 with X exponent bits (e.g. ``'fp8_e4'``)
    - ``'fp8_e4m3'`` / ``'fp8_e5m2'`` -> aliases for ``fp8_e4`` / ``fp8_e5``
    - an integer X in 0-7: the format with X exponent bits (0 is ``INT8``)
    """
    if isinstance(dtype, ElementFormat):
        if dtype in _EIGHT_BIT:
            return dtype
        raise ValueError(f"Invalid dtype {dtype!r}: fully_connected takes an 8-bit format.")
    if isinstance(dtype, int):
        if 0 <= dtype <= 7:
            return _EIGHT_BIT[dtype]
        raise ValueError(f"Invalid dtype {dtype!r}: exponent bits must be 0-7.")
    s = dtype.lower().strip()
    if s in ("int8", "fp8_e0"):
        return ElementFormat.INT8
    if s in ("fp8_e4m3", "e4m3"):
        return ElementFormat.E4
    if s in ("fp8_e5m2", "e5m2"):
        return ElementFormat.E5
    m = re.fullmatch(r"(?:fp8_)?e([1-7])", s)
    if m:
        return _EIGHT_BIT[int(m.group(1))]
    raise ValueError(f"Invalid dtype '{dtype}'. Use 'int8', 'fp8_eX', 'fp8_e4m3', or 'fp8_e5m2'.")


def _expand_int8_row(row: bytes | bytearray) -> bytes:
    """Expand 128 signed INT8 values into one 512-byte INT32 XMEM row."""
    signed = (value if value < 128 else value - 256 for value in row)
    return struct.pack("<128i", *signed)


def _load_inputs(state: IpuState, inputs_path: str | Path) -> None:
    """Load inputs with the row spacing required by the active mode."""
    if state.fmt is not ElementFormat.INT32:
        load_binary_to_xmem(state, inputs_path, INPUT_BASE_ADDR, INPUT_NEURONS, SAMPLES_NUM)
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
        raise ValueError(f"Weights file too small: {len(raw)} bytes, expected {expected}")

    original: list[bytes] = []
    for j in range(OUTPUT_NEURONS):
        row_start = j * INPUT_NEURONS
        original.append(raw[row_start : row_start + INPUT_NEURONS])

    for i in range(INPUT_NEURONS):
        transposed_vector = bytearray(INPUT_NEURONS)
        for j in range(OUTPUT_NEURONS):
            transposed_vector[j] = original[j][i]
        if state.fmt is ElementFormat.INT32:
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
        dtype:        The 8-bit data format (see :func:`parse_dtype`).
        wide_mode:    Run INT8 data in the emulator-only INT32 format.
    """

    def __init__(
        self, *, dtype: str | int | ElementFormat = "INT8", wide_mode: bool = False, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.inputs_path = Path(self.inputs_path)
        self.weights_path = Path(self.weights_path)
        self.dtype = parse_dtype(dtype)
        self.wide_mode = wide_mode
        SPEC.guard(dtype=self.dtype, wide_mode=wide_mode)

    @property
    def fmt(self) -> ElementFormat:
        """The element format this harness runs in: INT32 in wide mode, else its dtype."""
        return ElementFormat.INT32 if self.wide_mode else self.dtype

    def setup(self, state: IpuState) -> None:
        if state.fmt is ElementFormat.INT32:
            if self.dtype is not ElementFormat.INT8:
                raise ConfigurationError("fully-connected wide mode temporarily supports INT8 only")
        elif state.fmt is not self.dtype:
            raise ConfigurationError(
                f"fully-connected with dtype {self.dtype} needs a state in {self.fmt}, "
                f"not {state.fmt}"
            )

        row_size = WIDE_ROW_SIZE_BYTES if state.fmt is ElementFormat.INT32 else ROW_SIZE_BYTES
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
        # cr7 is the BLT.S lr0/lr1 loop bound: lr0 now increments by 1 row per
        # sample (was 128 bytes), so the bound is SAMPLES_NUM rows, not the
        # old byte count (10 * 128 = 1280).
        state.regfile.set_cr(6, 0)
        state.regfile.set_cr(7, SAMPLES_NUM)
        state.regfile.set_cr(8, 0)
        # Pre-decremented by one VLIW step because ADD runs before LDR/MULT within a cycle.
        # The 32-bit wraparound on the first ADD gives the correct starting values (0 for
        # both lr4 offset and lr5 element index on the first effective iteration).
        state.regfile.set_cr(
            9, (-1) & LR_CR_SCALAR_VALUE_MASK
        )  # lr4 cyclic offset (rows): pre-decremented
        state.regfile.set_cr(10, (-1) & LR_CR_SCALAR_VALUE_MASK)  # lr5 counter: pre-decremented
        state.regfile.set_cr(11, 127)  # BNE exit condition
        state.regfile.set_cr(12, 0)
        state.regfile.set_cr(13, WEIGHTS_BASE_ADDR // row_size)

    def teardown(self, state: IpuState) -> None:
        if self.output_path is not None:
            # ST.ACC writes each sample 512 B apart (OUTPUT_STRIDE_ROWS
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


def _supports(**params):
    if tuple(params.get("shape", (10, 128))) != (10, 128):
        return no("fully_connected requires input shape (10, 128)")
    if tuple(params.get("weight_shape", (64, 128))) != (64, 128):
        return no("fully_connected requires weight shape (64, 128)")
    try:
        dtype = parse_dtype(params.get("dtype", "INT8"))
    except (ValueError, TypeError, AttributeError) as exc:
        return no(str(exc))
    if params.get("wide_mode", False) and dtype is not ElementFormat.INT8:
        return no("wide_mode requires INT8")
    return yes()


SPEC = folder_spec(
    FullyConnectedApp,
    op="fully_connected",
    execution=lambda app: ExecutionConfig(fmt=app.fmt),
    supports=_supports,
    build=lambda **p: {"dtype": p.get("dtype", "INT8"), "wide_mode": p.get("wide_mode", False)},
    explain=lambda **_: "Fixed 10 x 128 inputs and 64 x 128 weights",
)
