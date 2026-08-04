"""Fully-connected layer test harness.

Computes, for each of SAMPLES_NUM input vectors::

    out[s, j] = sum_i in[s, i] * W[j, i]     j in [0, OUTPUT_NEURONS)

The weight matrix is transposed on staging so that element i of every output
neuron sits in one XMEM row, which is what ``LDR_CYCLIC_MULT_REG`` streams into
r_cyclic while ``MULT.RC.VE`` broadcasts in[s, i] from r0.

Usage::

    from ipu_apps.fully_connected import FullyConnectedApp

    app = FullyConnectedApp(
        inst_path="fc.bin",
        inputs_path="inputs.bin",
        weights_path="weights.bin",
        output_path="output.bin",
    )
    state, cycles = app.run()
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ipu_emu.ipu_config import LR_CR_SCALAR_VALUE_MASK

from ipu_apps.base import IpuApp

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

# -- Dimensions -------------------------------------------------------------

SAMPLES_NUM    = 10
INPUT_NEURONS  = 128   # contraction width (one full SIMD vector)
OUTPUT_NEURONS = 64

# ---------------------------------------------------------------------------
# Wide-vector FP32 only. Elements are 4 bytes and an XMEM row is LANES * 4 =
# 512 B, unconditionally -- there is no narrow path. INT8 is not a mode this
# kernel is written against; it belongs at the XMEM write boundary
# (ACTIVATE.QUANTIZE), which is what makes it invisible to the kernel.
#
# XMEM .asm operands are ROW numbers, not byte addresses (issue #179), and a
# row is LANES *elements*. Region bases are DERIVED from row counts rather
# than hardcoded as bytes: a hardcoded byte map sized for 1-byte elements
# overflows at 4 bytes/element and regions silently overwrite each other.
# ---------------------------------------------------------------------------
ELEM_BYTES = 4                               # FP32
LANES      = 128                             # elements per XMEM row
ROW_BYTES  = LANES * ELEM_BYTES              # 512

INPUT_STRIDE_ROWS   = 1     # one sample = one row of INPUT_NEURONS elements
WEIGHTS_STRIDE_ROWS = 1     # one transposed weight vector = one row
# One accumulator store writes all 512 B of r_acc. In wide mode a row is also
# 512 B, so a store is exactly one row and one sample's output owns one row.
OUTPUT_STRIDE_ROWS  = 1

INPUT_ROWS   = SAMPLES_NUM * INPUT_STRIDE_ROWS
WEIGHT_ROWS  = INPUT_NEURONS * WEIGHTS_STRIDE_ROWS

# inputs / weights / output packed back to back, in rows.
INPUT_BASE_ROW   = 0
WEIGHTS_BASE_ROW = INPUT_BASE_ROW + INPUT_ROWS
OUTPUT_BASE_ROW  = WEIGHTS_BASE_ROW + WEIGHT_ROWS

# Byte addresses for this harness's direct xmem staging (which bypasses row
# translation); the CR/LR values below stay in rows.
INPUT_BASE_ADDR   = INPUT_BASE_ROW * ROW_BYTES
WEIGHTS_BASE_ADDR = WEIGHTS_BASE_ROW * ROW_BYTES
OUTPUT_BASE_ADDR  = OUTPUT_BASE_ROW * ROW_BYTES


def _load_inputs(state: "IpuState", inputs_path: str | Path) -> None:
    """Stage the input activations, one sample per row.

    File layout: SAMPLES_NUM x INPUT_NEURONS FP32 values.
    """
    raw = Path(inputs_path).read_bytes()
    expected = SAMPLES_NUM * INPUT_NEURONS * ELEM_BYTES
    if len(raw) < expected:
        raise ValueError(
            f"Inputs file too small: {len(raw)} bytes, expected {expected}"
        )
    for s in range(SAMPLES_NUM):
        row = raw[s * INPUT_NEURONS * ELEM_BYTES : (s + 1) * INPUT_NEURONS * ELEM_BYTES]
        padded = bytearray(ROW_BYTES)
        padded[: len(row)] = row
        state.xmem.write_address(
            INPUT_BASE_ADDR + s * INPUT_STRIDE_ROWS * ROW_BYTES, padded
        )


def _load_and_transpose_weights(state: "IpuState", weights_path: str | Path) -> None:
    """Stage W transposed: row i holds W[:, i], i.e. element i of every neuron.

    File layout: OUTPUT_NEURONS x INPUT_NEURONS FP32 values (W[j][i]).
    XMEM layout: row i = [W[0][i], W[1][i], ... W[63][i], 0 ...].
    """
    raw = Path(weights_path).read_bytes()
    expected = OUTPUT_NEURONS * INPUT_NEURONS * ELEM_BYTES
    if len(raw) < expected:
        raise ValueError(
            f"Weights file too small: {len(raw)} bytes, expected {expected}"
        )
    w = np.frombuffer(raw[:expected], dtype=np.float32).reshape(
        OUTPUT_NEURONS, INPUT_NEURONS
    )
    for i in range(INPUT_NEURONS):
        padded = np.zeros(LANES, dtype=np.float32)
        padded[:OUTPUT_NEURONS] = w[:, i]
        state.xmem.write_address(
            WEIGHTS_BASE_ADDR + i * WEIGHTS_STRIDE_ROWS * ROW_BYTES,
            bytearray(padded.tobytes()),
        )


class FullyConnectedApp(IpuApp):
    """Fully-connected layer application harness (wide FP32).

    Args:
        inst_path:    Path to assembled instruction binary.
        inputs_path:  Path to input activations binary.
        weights_path: Path to weights binary.
        output_path:  Optional path to write output.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.inputs_path = Path(self.inputs_path)
        self.weights_path = Path(self.weights_path)

    def setup(self, state: "IpuState") -> None:
        _load_inputs(state, self.inputs_path)
        _load_and_transpose_weights(state, self.weights_path)
        # CR0=0 permanently (INPUT_BASE_ROW is 0, so nothing to set).
        # CR1=1 permanently (can't hold WEIGHTS_BASE_ROW; it lives on CR13).
        state.regfile.set_cr(2, OUTPUT_BASE_ROW)
        # Stride constants for assembly `add lrX lrX crN;;` (replacing removed `incr`).
        # lr0/lr14 walk one row (one input/weight vector) per iteration.
        state.regfile.set_cr(3, 1)
        state.regfile.set_cr(4, 1)
        state.regfile.set_cr(5, OUTPUT_STRIDE_ROWS)
        # Constants for ``SET lr* cr*`` in ``fully_connected.asm`` (issue #82).
        # cr7 is the BLT lr0/lr1 loop bound: lr0 increments by 1 row per
        # sample, so the bound is SAMPLES_NUM rows.
        state.regfile.set_cr(6, 0)
        state.regfile.set_cr(7, SAMPLES_NUM)
        state.regfile.set_cr(8, 0)
        # Pre-decremented by one VLIW step because ADD runs before LDR/MULT within a cycle.
        # The 32-bit wraparound on the first ADD gives the correct starting values (0 for
        # both lr4 offset and lr5 element index on the first effective iteration).
        state.regfile.set_cr(9, (-1) & LR_CR_SCALAR_VALUE_MASK)     # lr4 cyclic offset (rows): pre-decremented
        state.regfile.set_cr(10, (-1) & LR_CR_SCALAR_VALUE_MASK)    # lr5 counter: pre-decremented
        state.regfile.set_cr(11, INPUT_NEURONS - 1)  # BNE exit condition
        state.regfile.set_cr(12, 0)
        state.regfile.set_cr(13, WEIGHTS_BASE_ROW)   # moved from cr1 (CR1 is now locked)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            # One sample's output per row; only the first OUTPUT_NEURONS lanes
            # are real -- dump_xmem_to_binary can't express a stride different
            # from its chunk size, so read samples individually.
            sample_stride = OUTPUT_STRIDE_ROWS * ROW_BYTES
            sample_size = OUTPUT_NEURONS * ELEM_BYTES
            parts = [
                bytes(state.xmem.read_address(
                    OUTPUT_BASE_ADDR + i * sample_stride, sample_size
                ))
                for i in range(SAMPLES_NUM)
            ]
            Path(self.output_path).write_bytes(b"".join(parts))
