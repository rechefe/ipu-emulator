"""Transformer matmul 720×240×128 harness (Layer 5 QKV, 16 tokens).

Computes C[j, t] = sum_k W[j, k] * D[k, t]  for all j in [0, 720), t in [0, 16).

  D: channel-major [240, 16] input  — K channels × 16 tokens
  W: output-major  [720, 240] weights — N_OUT rows × K cols, stored verbatim (no transpose)
  C: channel-major [720, 16] output  — N_OUT channels × 16 tokens (FP32 accumulators)

Single token group: 16 tokens fit in one 128-lane SIMD vector, so each output
channel needs only one accumulate-and-store pass.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ipu_emu.emulator import dump_xmem_to_binary

from ipu_apps.base import IpuApp

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

# -- Dimensions -------------------------------------------------------------

K      = 240   # input channels
N_OUT  = 720   # output channels
N_TOK  = 16    # tokens (single group, padded to LANES in XMEM)

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

W_STRIDE_ROWS    = -(-K // LANES)   # weight chunks (rows) per output channel: 2
DATA_STRIDE_ROWS = 1                # one row per input channel (N_TOK padded to LANES)

# One accumulator store writes all 512 B of r_acc. In wide mode a row is also
# 512 B, so a store is exactly one row and one output channel owns one row.
# Per kernel_layer_map.md's crop convention, a PRODUCER emits full, uncropped
# rows -- only the FINAL consumer in a chain crops. teardown() below dumps the
# full row; a caller that wants the densely-packed N_OUT x N_TOK form crops it
# itself (this used to crop here, which silently broke residual_add_16x240's
# full-row input contract -- see kernel_docs/kernel_layer_map.md).
OUTPUT_ROW_BYTES   = 512
OUTPUT_STRIDE_ROWS = 1

DATA_ROWS   = K * DATA_STRIDE_ROWS
WEIGHT_ROWS = N_OUT * W_STRIDE_ROWS

# D/W/C packed back to back, in rows.
DATA_BASE_ROW    = 0
WEIGHTS_BASE_ROW = DATA_BASE_ROW + DATA_ROWS
OUTPUT_BASE_ROW  = WEIGHTS_BASE_ROW + WEIGHT_ROWS

# Byte addresses for this harness's direct xmem staging (which bypasses row
# translation); the CR/LR values below stay in rows.
DATA_BASE    = DATA_BASE_ROW * ROW_BYTES
WEIGHTS_BASE = WEIGHTS_BASE_ROW * ROW_BYTES
OUTPUT_BASE  = OUTPUT_BASE_ROW * ROW_BYTES


def _load_data(state: "IpuState", data_path: str | Path) -> None:
    """Stage channel-major D, padding each channel's N_TOK elements to a row.

    File layout: K channels × N_TOK elements (D[k][tok] at k*N_TOK + tok).
    XMEM layout: channel k at row DATA_BASE_ROW + k (N_TOK valid + zero pad).
    """
    raw = Path(data_path).read_bytes()
    expected = K * N_TOK * ELEM_BYTES
    if len(raw) < expected:
        raise ValueError(f"{data_path}: expected >= {expected} B, got {len(raw)}")
    for k in range(K):
        buf = bytearray(ROW_BYTES)
        buf[: N_TOK * ELEM_BYTES] = raw[
            k * N_TOK * ELEM_BYTES : (k * N_TOK + N_TOK) * ELEM_BYTES
        ]
        state.xmem.write_address(DATA_BASE + k * DATA_STRIDE_ROWS * ROW_BYTES, buf)


def _load_weights(state: "IpuState", weights_path: str | Path) -> None:
    """Stage output-major W (no transposition), one LANES-element chunk per row.

    File layout: W[j][k] at element j*K + k  (N_OUT rows × K cols).
    XMEM layout: chunk c of channel j at row WEIGHTS_BASE_ROW + j*W_STRIDE_ROWS + c,
    the final chunk zero-padded out to a whole row (K is not a multiple of LANES).
    """
    raw = Path(weights_path).read_bytes()
    expected = N_OUT * K * ELEM_BYTES
    if len(raw) < expected:
        raise ValueError(f"{weights_path}: expected >= {expected} B, got {len(raw)}")
    for j in range(N_OUT):
        row = raw[j * K * ELEM_BYTES : (j * K + K) * ELEM_BYTES]
        for c in range(W_STRIDE_ROWS):
            lo = c * LANES
            hi = min(lo + LANES, K)
            buf = bytearray(ROW_BYTES)
            buf[: (hi - lo) * ELEM_BYTES] = row[lo * ELEM_BYTES : hi * ELEM_BYTES]
            state.xmem.write_address(
                WEIGHTS_BASE + (j * W_STRIDE_ROWS + c) * ROW_BYTES, buf
            )


class MatMul720x240x128App(IpuApp):
    """720x240_x128 transformer matmul harness (Layer 5 QKV)."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)
        self.weights_path = Path(self.weights_path)

    def setup(self, state: "IpuState") -> None:
        _load_data(state, self.input_path)
        _load_weights(state, self.weights_path)
        # CR1 (≡1) is a read-only hardwired constant; WEIGHTS_BASE lives in CR9.
        state.regfile.set_cr(0, DATA_BASE_ROW)
        state.regfile.set_cr(9, WEIGHTS_BASE_ROW)
        state.regfile.set_cr(2, WEIGHTS_BASE_ROW + 1)   # next weight row
        state.regfile.set_cr(5, OUTPUT_BASE_ROW)
        state.regfile.set_cr(6, -DATA_STRIDE_ROWS)             # data startup (rows)
        state.regfile.set_cr(8, -1)                            # per-chunk fixed_idx startup
        state.regfile.set_lr(0, 0)                             # r_cyclic write-index 0
        state.regfile.set_lr(2, DATA_STRIDE_ROWS)              # data stride (rows)
        state.regfile.set_lr(3, OUTPUT_STRIDE_ROWS)            # output stride (1 row/channel)
        state.regfile.set_lr(6, 126)                           # width-128 chunk bound
        state.regfile.set_lr(7, 0)                             # output pointer
        state.regfile.set_lr(8, 0)                             # weight row offset
        state.regfile.set_lr(9, 0)                             # j counter
        state.regfile.set_lr(10, N_OUT)                        # j-loop limit
        state.regfile.set_lr(11, 110)                          # tail-chunk bound: width=112
        state.regfile.set_lr(12, W_STRIDE_ROWS)                # weight stride per j (rows)

    def teardown(self, state: "IpuState") -> None:
        """Dump full, uncropped rows -- see the crop-convention note above
        OUTPUT_ROW_BYTES. Callers that need the densely-packed N_OUT x N_TOK
        form crop it themselves (see the wide test for the pattern)."""
        if self.output_path is not None:
            dump_xmem_to_binary(
                state, self.output_path,
                OUTPUT_BASE, OUTPUT_ROW_BYTES, N_OUT,
            )
