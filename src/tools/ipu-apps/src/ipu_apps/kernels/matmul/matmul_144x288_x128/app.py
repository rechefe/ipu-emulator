"""Transformer matmul 144×288 harness (FFN linear 2).

Computes C[j, t] = sum_k W[j, k] * D[k, t]  for all j in [0, 144), t in [0, 256).

Usage::

    from ipu_apps.kernels.matmul.matmul_144x288_x128.app import MatMul144x288x128App
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ipu_emu.emulator import dump_xmem_to_binary

from ipu_apps.kernel_registry.base import IpuApp
from ipu_apps.kernels.matmul.app import matmul_spec, streamed_data_tables

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

K     = 288
N_OUT = 144
N_TG  = 2
N_TOK = 128

# ---------------------------------------------------------------------------
# Wide-vector FP32 only. Elements are 4 bytes and an XMEM row is LANES * 4 =
# 512 B, unconditionally -- there is no narrow path here. INT8 is not a mode
# this kernel is written against; it belongs at the XMEM write boundary
# (ACTIVATE.QUANTIZE), which is what makes it invisible to the kernel.
#
# XMEM .asm operands are ROW numbers, not byte addresses, and a
# row is LANES *elements*. Region bases are DERIVED from row counts rather
# than hardcoded as bytes: a hardcoded byte map sized for 1-byte elements
# overflows at 4 bytes/element and silently corrupts the run.
# ---------------------------------------------------------------------------
ELEM_BYTES = 4                               # FP32
LANES      = 128                             # elements per XMEM row
ROW_BYTES  = LANES * ELEM_BYTES              # 512

W_STRIDE_ROWS    = -(-K // LANES)            # rows per output channel (ceil) = 3
DATA_STRIDE_ROWS = (N_TG * N_TOK) // LANES   # rows per input channel
W_STRIDE         = W_STRIDE_ROWS * LANES     # elements per output channel (padded)

# One accumulator store writes all LANES elements of R_ACC -- exactly one row
# in wide mode, so one output channel owns one row.
OUTPUT_ROW_BYTES   = 512
OUTPUT_STRIDE_ROWS = 1

DATA_ROWS   = K * N_TG                       # one row per (k, tg)
WEIGHT_ROWS = N_OUT * W_STRIDE_ROWS

# D/W/C packed back to back, in rows.
DATA_BASE_ROW    = 0
WEIGHTS_BASE_ROW = DATA_BASE_ROW + DATA_ROWS + DATA_STRIDE_ROWS  # + overhead: the last prefetch's row
OUTPUT_BASE_ROW  = WEIGHTS_BASE_ROW + WEIGHT_ROWS

# Byte addresses for this harness's direct xmem staging (which bypasses row
# translation); the CR/LR values below stay in rows.
DATA_BASE    = DATA_BASE_ROW * ROW_BYTES
WEIGHTS_BASE = WEIGHTS_BASE_ROW * ROW_BYTES
OUTPUT_BASE  = OUTPUT_BASE_ROW * ROW_BYTES


def _load_data(state: "IpuState", data_path: str | Path) -> None:
    """Stage D. Channel-major and already contiguous, so a straight copy works.

    File layout: K=288 channels × N_TG tg × N_TOK elements each.
    """
    raw = Path(data_path).read_bytes()
    expected = K * N_TG * N_TOK * ELEM_BYTES
    if len(raw) < expected:
        raise ValueError(f"{data_path}: expected >= {expected} B, got {len(raw)}")
    state.xmem.write_address(DATA_BASE, bytearray(raw[:expected]))


def _load_weights(state: "IpuState", weights_path: str | Path) -> None:
    """Stage W, padding each output channel's K elements out to whole rows.

    File layout: W[j][k] at element j*K + k  (N_OUT rows × K=288 cols).
    XMEM layout per output channel j (3 rows):
      row 0: W[j, 0..127]
      row 1: W[j, 128..255]
      row 2: W[j, 256..287] + 96 zero elements
    """
    raw = Path(weights_path).read_bytes()
    row_elems = LANES                      # elements per XMEM row
    stride = W_STRIDE_ROWS * row_elems     # elements per output channel (padded)
    for j in range(N_OUT):
        row = raw[j * K * ELEM_BYTES : (j * K + K) * ELEM_BYTES]
        for chunk in range(W_STRIDE_ROWS):
            lo = chunk * row_elems
            hi = min(lo + row_elems, K)
            buf = bytearray(row_elems * ELEM_BYTES)
            if hi > lo:
                buf[: (hi - lo) * ELEM_BYTES] = row[lo * ELEM_BYTES : hi * ELEM_BYTES]
            state.xmem.write_address(
                WEIGHTS_BASE + (j * stride + lo) * ELEM_BYTES, buf
            )


class MatMul144x288x128App(IpuApp):
    """144×288 transformer matmul application harness."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)
        self.weights_path = Path(self.weights_path)

    def setup(self, state: "IpuState") -> None:
        _load_data(state, self.input_path)
        _load_weights(state, self.weights_path)

        # CR1 (≡1) is a read-only hardwired constant —
        # writing anything else raises EmulatorError. WEIGHTS_BASE lives on CR9 (free).
        # DATA_BASE gets its own table CR, CR12 -- never CR0, even at row 0. CR2/CR3 are writable.
        state.regfile.set_cr(12, DATA_BASE_ROW)
        state.regfile.set_cr(9, WEIGHTS_BASE_ROW)
        state.regfile.set_cr(2, W_STRIDE_ROWS - 1)              # LAST_CHUNK: tg=1 back to chunk 0
        state.regfile.set_cr(4, OUTPUT_BASE_ROW)                                    # tg=0 output
        state.regfile.set_cr(5, OUTPUT_BASE_ROW + N_OUT * OUTPUT_STRIDE_ROWS)       # tg=1 output
        state.regfile.set_cr(6, -DATA_STRIDE_ROWS)              # tg=0 data startup (rows)
        state.regfile.set_cr(7, -(DATA_STRIDE_ROWS // N_TG))    # tg=1 data startup (rows)
        state.regfile.set_cr(8, -1)                             # per-chunk fixed_idx startup
        state.regfile.set_lr(0, 0)                              # R_CYCLIC write-index 0
        state.regfile.set_lr(2, DATA_STRIDE_ROWS)               # data stride (rows)
        state.regfile.set_lr(3, OUTPUT_STRIDE_ROWS)             # output stride (rows)
        state.regfile.set_lr(6, 126)                            # per-chunk bound: first_index=0, width=128 → 126
        state.regfile.set_lr(7, 0)                              # output pointer
        state.regfile.set_lr(8, 0)                              # weight row offset
        state.regfile.set_lr(9, 0)                              # j counter
        state.regfile.set_lr(10, N_OUT)                         # j-loop limit (144)
        # K=288 is NOT a multiple of LANES: the chunks are 128 + 128 + 32, so the
        # last chunk needs its own narrower bound (first_index=0, width=32 → 30).
        # A width-128 bound there would run the tail 96 steps into zero padding.
        state.regfile.set_lr(11, (K % LANES) - 2)               # tail-chunk bound (width 32 → 30)
        state.regfile.set_lr(12, 1)                             # weight stride per j: 1 row past the last chunk

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            dump_xmem_to_binary(
                state, self.output_path,
                OUTPUT_BASE, OUTPUT_ROW_BYTES, N_OUT * N_TG,
            )


# The registry's (M, K, N): M is the total token count, N the output channels.
M = N_TG * N_TOK
N = N_OUT

def _tables(**params):
    """XMEM tables as the Cache Unit sees them (see ``ipu_emu.table_trace``)."""
    return streamed_data_tables(
        n_out=N_OUT, data_rows=DATA_ROWS, data_stride=DATA_STRIDE_ROWS, walks=N_OUT * N_TG,
        chunks=3, reread_chunks=True, outputs={'output_tg0': 4, 'output_tg1': 5})


SPEC = matmul_spec(MatMul144x288x128App, m=M, k=K, n=N, role="144x288 transformer matmul", tables=_tables)
