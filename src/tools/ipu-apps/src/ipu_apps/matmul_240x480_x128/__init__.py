"""Transformer matmul 240×480×128 harness (Layer 5 FFN2, 16 tokens).

Computes C[j, t] = sum_k W[j, k] * D[k, t]  for all j in [0, 240), t in [0, 16).

  D: channel-major [480, 16] input  — K channels × 16 tokens
  W: output-major  [240, 480] weights — N_OUT rows × K cols, stored verbatim (no transpose)
  C: channel-major [240, 16] output  — N_OUT channels × 16 tokens (FP32 accumulators)

Single token group: 16 tokens fit in one 128-lane SIMD vector, so each output
channel needs only one accumulate-and-store pass.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ipu_emu.ipu_math import DType
from ipu_emu.emulator import dump_xmem_to_binary

from ipu_apps.base import IpuApp

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

# -- Dimensions -------------------------------------------------------------

K      = 480   # input channels
N_OUT  = 240   # output channels
N_TOK  = 16    # tokens (single group, padded to 128 in XMEM)
N_LANE = 128   # SIMD lanes / padded tokens per channel

# -- Memory map -------------------------------------------------------------

DATA_BASE    = 0x00000
WEIGHTS_BASE = 0x10000
OUTPUT_BASE  = 0x30000

W_STRIDE         = 512    # bytes per output channel in XMEM (ceil(K/128)*128)
DATA_STRIDE      = N_LANE      # bytes per channel in XMEM (128, padded)
OUTPUT_ROW_BYTES = N_TOK * 4   # bytes per output channel (packed)

# XMEM .asm operands are ROW numbers (one row = 128 elements), not byte
# addresses -- see issue #179. The *_BASE constants above stay byte addresses
# because they only drive the harness's direct xmem read/write calls (which
# bypass row translation); the CR/LR registers below feed the .asm's XMEM
# instructions instead, so they carry addresses and strides converted to rows.
ROW_SIZE_BYTES = 128
DATA_BASE_ROW    = DATA_BASE // ROW_SIZE_BYTES
WEIGHTS_BASE_ROW = WEIGHTS_BASE // ROW_SIZE_BYTES
OUTPUT_BASE_ROW  = OUTPUT_BASE // ROW_SIZE_BYTES
W_STRIDE_ROWS    = W_STRIDE // ROW_SIZE_BYTES
DATA_STRIDE_ROWS = DATA_STRIDE // ROW_SIZE_BYTES         # 1 row per channel
# One output channel per XMEM row. OUTPUT_ROW_BYTES (N_TOK*4 = 64 B) is the
# MEANINGFUL prefix of each row, not the store stride: STR_ACC_REG always
# writes all 512 B of r_acc, and a row belongs to exactly one output channel --
# rows are never shared between channels. So channel j owns row j, holding
# 64 B of real output followed by unused tail.
#
# The pre-#179 kernel used a 64 B store stride so consecutive channels packed
# two-per-row, each store burying the previous one's tail. That straddled row
# boundaries, which the row-addressed model does not permit (and a 64 B stride
# is not even expressible: addresses are row*128). teardown() now crops each
# row's 64 B prefix, so the on-disk output is byte-identical to before.
OUTPUT_STRIDE_ROWS = 1

_DTYPE_MAP = {
    "INT8":   DType.INT8,
    "int8":   DType.INT8,
    "E4":     DType.E4,
    "fp8_e4": DType.E4,
    "E5":     DType.E5,
    "fp8_e5": DType.E5,
}


def parse_dtype(dtype_str: str) -> DType:
    dt = _DTYPE_MAP.get(dtype_str)
    if dt is None:
        raise ValueError(f"Invalid dtype '{dtype_str}'. Supported: INT8, E4, E5")
    return dt


def _load_data(state: "IpuState", data_path: str | Path) -> None:
    """Load channel-major input into XMEM, padding each channel to 128 bytes.

    File layout: K channels × N_TOK bytes (D[k][tok] at k*N_TOK + tok).
    XMEM layout:  channel k at DATA_BASE + k*128 (N_TOK valid + zero pad).
    """
    raw = Path(data_path).read_bytes()
    for k in range(K):
        row = raw[k * N_TOK : k * N_TOK + N_TOK]
        padded = bytearray(N_LANE)
        padded[:N_TOK] = row
        state.xmem.write_address(DATA_BASE + k * DATA_STRIDE, padded)


def _load_weights(state: "IpuState", weights_path: str | Path) -> None:
    """Load output-major weights into XMEM (no transposition), 4 chunks of 128 bytes.

    File layout: W[j][k] at byte j*K + k  (N_OUT rows × K cols).
    """
    raw = Path(weights_path).read_bytes()
    for j in range(N_OUT):
        row = raw[j * K : j * K + K]
        state.xmem.write_address(WEIGHTS_BASE + j * W_STRIDE + 0, bytearray(row[0:128]))
        state.xmem.write_address(WEIGHTS_BASE + j * W_STRIDE + 128, bytearray(row[128:256]))
        state.xmem.write_address(WEIGHTS_BASE + j * W_STRIDE + 256, bytearray(row[256:384]))
        chunk3 = bytearray(128)
        chunk3[:96] = row[384:480]
        state.xmem.write_address(WEIGHTS_BASE + j * W_STRIDE + 384, chunk3)


class MatMul240x480x128App(IpuApp):
    """240x480_x128 transformer matmul harness (Layer 5 FFN2)."""

    def __init__(self, *, dtype: str | DType = "INT8", **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)
        self.weights_path = Path(self.weights_path)
        self.dtype = parse_dtype(dtype) if isinstance(dtype, str) else dtype

    def setup(self, state: "IpuState") -> None:
        state.dtype = self.dtype
        _load_data(state, self.input_path)
        _load_weights(state, self.weights_path)
        # CR1 (≡1) is a read-only hardwired constant; WEIGHTS_BASE lives in CR9.
        state.regfile.set_cr(0, DATA_BASE_ROW)
        state.regfile.set_cr(9, WEIGHTS_BASE_ROW)
        state.regfile.set_cr(2, WEIGHTS_BASE_ROW + 1)   # next weight row
        state.regfile.set_cr(3, WEIGHTS_BASE_ROW + 2)   # +2 weight rows
        state.regfile.set_cr(4, WEIGHTS_BASE_ROW + 3)   # +3 weight rows
        state.regfile.set_cr(5, OUTPUT_BASE_ROW)
        state.regfile.set_cr(6, -DATA_STRIDE_ROWS)                  # data startup: -128
        state.regfile.set_cr(8, -1)                            # per-chunk fixed_idx startup
        state.regfile.set_lr(0, 0)                             # r_cyclic write-index 0
        state.regfile.set_lr(2, DATA_STRIDE_ROWS)                   # data stride (128)
        state.regfile.set_lr(3, OUTPUT_STRIDE_ROWS)            # output stride (1 row/channel)
        state.regfile.set_lr(6, 126)                           # width-128 chunk bound
        state.regfile.set_lr(7, 0)                             # output pointer
        state.regfile.set_lr(8, 0)                             # weight byte offset
        state.regfile.set_lr(9, 0)                             # j counter
        state.regfile.set_lr(10, N_OUT)                        # j-loop limit
        state.regfile.set_lr(11, 94)                           # tail-chunk bound: width=96
        state.regfile.set_lr(12, W_STRIDE_ROWS)                     # weight stride per j

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            # Each output channel owns one XMEM row: OUTPUT_ROW_BYTES of real
            # data followed by unused tail. dump_xmem_to_binary cannot express a
            # stride different from its chunk size, so read the rows
            # individually and concatenate their prefixes -- this reproduces the
            # packed layout the golden expects without ever sharing a row
            # between two channels.
            row_stride = OUTPUT_STRIDE_ROWS * ROW_SIZE_BYTES
            parts = [
                bytes(state.xmem.read_address(OUTPUT_BASE + j * row_stride,
                                              OUTPUT_ROW_BYTES))
                for j in range(N_OUT)
            ]
            Path(self.output_path).write_bytes(b"".join(parts))
