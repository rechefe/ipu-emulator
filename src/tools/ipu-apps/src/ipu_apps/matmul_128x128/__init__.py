"""Matrix-multiplication 128×128 test harness.

Computes C = A × W^T where:
  A: (M=128, K=128) input matrix  — row-major, one XMEM row per matrix row
  W: (N=128, K=128) weight matrix — output-major, W[n][k] (row n = all K inputs for output n)
  C: (M=128, N=128) output matrix — one XMEM row per matrix row (128 accumulators)

Weights are stored in file as W[n][k] (output-major, matching FC convention).
Python transposes W → T before loading: T[k] = column k of W = all N output weights for input k.

Usage::

    from ipu_apps.matmul_128x128 import MatMul128x128App

    app = MatMul128x128App(
        inst_path="matmul_128x128.bin",
        input_path="input.bin",
        weights_path="weights.bin",
        output_path="output.bin",
    )
    state, cycles = app.run()
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ipu_emu.emulator import dump_xmem_to_binary

from ipu_apps.base import IpuApp

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

# -- Dimensions -------------------------------------------------------------

M = 128   # rows of A / rows of C
K = 128   # cols of A / cols of W per row
N = 128   # rows of W (output neurons) / cols of C  (must equal SIMD width = 128)

# ---------------------------------------------------------------------------
# Wide-vector FP32 only. Elements are 4 bytes and an XMEM row is LANES * 4 =
# 512 B, unconditionally -- there is no narrow path here. INT8 is not a mode
# this kernel is written against; it belongs at the XMEM write boundary
# (ACTIVATE.QUANTIZE), which is what makes it invisible to the kernel.
#
# XMEM .asm operands are ROW numbers, not byte addresses (issue #179), and a
# row is LANES *elements*. Region bases are DERIVED from row counts rather
# than hardcoded as bytes: a hardcoded byte map sized for 1-byte elements
# overflows at 4 bytes/element and silently corrupts the run.
# ---------------------------------------------------------------------------
ELEM_BYTES = 4                    # FP32
LANES      = 128                  # elements per XMEM row
ROW_BYTES  = LANES * ELEM_BYTES   # 512

# A row (padded to LANES elements) per matrix row; T[k] is one row per k.
INPUT_ROWS   = M
WEIGHT_ROWS  = K

# One accumulator store writes all 512 B of r_acc. In wide mode a row is also
# 512 B, so a store is exactly one row and one output row of C owns one row.
OUTPUT_ROW_BYTES   = N * ELEM_BYTES   # 512
OUTPUT_STRIDE_ROWS = 1
VEC_STRIDE_ROWS    = 1                # one LANES-element vector = 1 row

# A / T / C packed back to back, in rows.
INPUT_BASE_ROW   = 0
WEIGHTS_BASE_ROW = INPUT_BASE_ROW + INPUT_ROWS
OUTPUT_BASE_ROW  = WEIGHTS_BASE_ROW + WEIGHT_ROWS

# Byte addresses for this harness's direct xmem staging (which bypasses row
# translation); the CR/LR values below stay in rows.
INPUT_BASE_ADDR   = INPUT_BASE_ROW * ROW_BYTES
WEIGHTS_BASE_ADDR = WEIGHTS_BASE_ROW * ROW_BYTES
OUTPUT_BASE_ADDR  = OUTPUT_BASE_ROW * ROW_BYTES


def _load_input(state: "IpuState", input_path: str | Path) -> None:
    """Stage A, one XMEM row per matrix row (K == LANES, so no padding)."""
    raw = Path(input_path).read_bytes()
    expected = M * K * ELEM_BYTES
    if len(raw) < expected:
        raise ValueError(f"{input_path}: expected >= {expected} B, got {len(raw)}")
    for m in range(M):
        buf = bytearray(ROW_BYTES)
        row = raw[m * K * ELEM_BYTES : (m * K + K) * ELEM_BYTES]
        buf[: len(row)] = row
        state.xmem.write_address(INPUT_BASE_ADDR + m * ROW_BYTES, buf)


def _load_and_transpose_weights(state: "IpuState", weights_path: str | Path) -> None:
    """Load W[n][k] from file and write T[k] (column k of W) into XMEM.

    File layout: W[n][k] at element n*K + k  (N rows × K cols, output-major).
    XMEM layout: T[k] at row WEIGHTS_BASE_ROW + k, padded to LANES elements.
    T[k][n] = W[n][k] = weight from input k to output n.
    """
    raw = Path(weights_path).read_bytes()
    expected = N * K * ELEM_BYTES
    if len(raw) < expected:
        raise ValueError(f"{weights_path}: expected >= {expected} B, got {len(raw)}")
    for k in range(K):
        t_row = bytearray(ROW_BYTES)
        for n in range(N):
            src = (n * K + k) * ELEM_BYTES
            t_row[n * ELEM_BYTES : (n + 1) * ELEM_BYTES] = raw[src : src + ELEM_BYTES]
        state.xmem.write_address(WEIGHTS_BASE_ADDR + k * ROW_BYTES, t_row)


class MatMul128x128App(IpuApp):
    """128×128 matrix-multiplication application harness.

    Args:
        inst_path:    Path to assembled instruction binary.
        input_path:   Path to input matrix A binary (M×K FP32, row-major).
        weights_path: Path to weight matrix W binary (N×K FP32, output-major W[n][k]).
        output_path:  Optional path to write output C.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)
        self.weights_path = Path(self.weights_path)

    def setup(self, state: "IpuState") -> None:
        _load_input(state, self.input_path)
        _load_and_transpose_weights(state, self.weights_path)
        # CR0 (≡0) and CR1 (≡1) are read-only hardwired constants on the new
        # architecture — writes are silently dropped. INPUT_BASE_ROW is 0, so
        # cr0 still reads the correct input base; the weights base is moved to
        # CR11 (a free CR) instead of CR1. See MIGRATION_CHECKLIST.md Bug #2.
        state.regfile.set_cr(0, INPUT_BASE_ROW)
        state.regfile.set_cr(11, WEIGHTS_BASE_ROW)
        state.regfile.set_cr(2, OUTPUT_BASE_ROW)
        state.regfile.set_cr(3, 1)
        state.regfile.set_cr(4, VEC_STRIDE_ROWS)                 # input/weight stride (1 row)
        state.regfile.set_cr(5, OUTPUT_STRIDE_ROWS)              # output stride (rows)
        state.regfile.set_cr(6, M * VEC_STRIDE_ROWS)             # outer-loop limit, compared vs row ptr
        state.regfile.set_cr(7, 0)
        state.regfile.set_cr(8, -VEC_STRIDE_ROWS)                # weight-offset startup (-1 row)
        state.regfile.set_cr(9, -1)
        state.regfile.set_cr(10, K - 1)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            dump_xmem_to_binary(
                state, self.output_path,
                OUTPUT_BASE_ADDR, OUTPUT_ROW_BYTES, M,
            )
