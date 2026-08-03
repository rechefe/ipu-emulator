"""Transformer matmul 144×144 harness.

Computes C[j, t] = sum_k W[j, k] * D[k, t]  for all j in [0, 144), t in [0, 256).

Usage::

    from ipu_apps.matmul_144x144_x128 import MatMul144x144x128App
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ipu_emu.ipu_math import DType
from ipu_emu.emulator import dump_xmem_to_binary

from ipu_apps.base import IpuApp

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

K     = 144
N_OUT = 144
N_TG  = 2
N_TOK = 128

DATA_BASE    = 0x00000
WEIGHTS_BASE = 0x10000
OUTPUT_BASE  = 0x20000

W_STRIDE        = 256
OUTPUT_ROW_BYTES = 512

# XMEM .asm operands are ROW numbers (one row = 128 elements), not byte
# addresses -- see issue #179. The *_BASE constants above stay byte addresses
# because they only drive direct state.xmem.write_address/read_address calls in
# this harness (which bypass row translation); the CR/LR registers below feed
# the .asm's XMEM instructions instead, so they carry the same addresses and
# strides converted to rows.
ROW_SIZE_BYTES = 128
DATA_BASE_ROW    = DATA_BASE // ROW_SIZE_BYTES
WEIGHTS_BASE_ROW = WEIGHTS_BASE // ROW_SIZE_BYTES
OUTPUT_BASE_ROW  = OUTPUT_BASE // ROW_SIZE_BYTES
W_STRIDE_ROWS    = W_STRIDE // ROW_SIZE_BYTES            # 2 rows per output channel
DATA_STRIDE_ROWS = (N_TG * N_TOK) // ROW_SIZE_BYTES      # 2 rows per input channel
# STR_ACC_REG always writes all 512 B of r_acc regardless of dtype, which spans
# 4 rows in narrow mode -- OUTPUT_ROW_BYTES already matches that full payload,
# so the output stride is exactly that store's footprint and rows never overlap.
OUTPUT_STRIDE_ROWS = OUTPUT_ROW_BYTES // ROW_SIZE_BYTES  # 4

_DTYPE_MAP = {
    "INT8":     DType.INT8,
    "int8":     DType.INT8,
    "E4": DType.E4,
    "fp8_e4": DType.E4,
    "E5": DType.E5,
    "fp8_e5": DType.E5,
}


def parse_dtype(dtype_str: str) -> DType:
    dt = _DTYPE_MAP.get(dtype_str)
    if dt is None:
        raise ValueError(f"Invalid dtype '{dtype_str}'. Supported: INT8, E4, E5")
    return dt


def _load_data(state: "IpuState", data_path: str | Path) -> None:
    raw = Path(data_path).read_bytes()
    state.xmem.write_address(DATA_BASE, bytearray(raw))


def _load_weights(state: "IpuState", weights_path: str | Path) -> None:
    raw = Path(weights_path).read_bytes()
    for j in range(N_OUT):
        row = raw[j * K : j * K + K]
        state.xmem.write_address(WEIGHTS_BASE + j * W_STRIDE, bytearray(row[:128]))
        tail = bytearray(128)
        tail[:K - 128] = row[128:K]
        state.xmem.write_address(WEIGHTS_BASE + j * W_STRIDE + 128, tail)


class MatMul144x144x128App(IpuApp):
    """144×144 transformer matmul application harness."""

    def __init__(self, *, dtype: str | DType = "INT8", **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)
        self.weights_path = Path(self.weights_path)
        self.dtype = parse_dtype(dtype) if isinstance(dtype, str) else dtype

    def setup(self, state: "IpuState") -> None:
        state.dtype = self.dtype
        _load_data(state, self.input_path)
        _load_weights(state, self.weights_path)
        # CR1 (≡1) is a read-only hardwired constant on the new architecture —
        # writes are silently dropped. WEIGHTS_BASE is moved to CR9 (free).
        # cr0=DATA_BASE is 0x0 (harmless no-op, matches hardwired 0); cr2 is a
        # writable CR and stays. See MIGRATION_CHECKLIST.md Bug #2.
        state.regfile.set_cr(0, DATA_BASE_ROW)
        state.regfile.set_cr(9, WEIGHTS_BASE_ROW)
        state.regfile.set_cr(2, WEIGHTS_BASE_ROW + 1)           # W[j,128..143]: next row
        state.regfile.set_cr(3, OUTPUT_BASE_ROW)                                    # tg=0 output
        state.regfile.set_cr(4, OUTPUT_BASE_ROW + N_OUT * OUTPUT_STRIDE_ROWS)       # tg=1 output
        state.regfile.set_cr(5, -DATA_STRIDE_ROWS)              # tg=0 data startup (rows)
        state.regfile.set_cr(6, -(DATA_STRIDE_ROWS // N_TG))    # tg=1 data startup (rows)
        state.regfile.set_cr(7, -1)                             # k-loop1 fixed_idx startup
        state.regfile.set_cr(8, 127)                            # k-loop2 fixed_idx startup
        state.regfile.set_lr(0, 0)                              # r_cyclic write-index 0
        state.regfile.set_lr(2, DATA_STRIDE_ROWS)               # data stride (rows)
        state.regfile.set_lr(3, OUTPUT_STRIDE_ROWS)             # output stride (rows)
        state.regfile.set_lr(6, 126)                            # k-loop1 bound: first_index=0, width=128 → 0+128-2=126
        state.regfile.set_lr(7, 0)                              # output pointer
        state.regfile.set_lr(8, 0)                              # weight byte offset
        state.regfile.set_lr(9, 0)                              # j counter
        state.regfile.set_lr(10, N_OUT)                         # j-loop limit
        state.regfile.set_lr(11, 142)                           # k-loop2 bound: first_index=128, width=16 → 128+16-2=142
        state.regfile.set_lr(12, W_STRIDE_ROWS)                 # weight stride per j (rows)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            dump_xmem_to_binary(
                state, self.output_path,
                OUTPUT_BASE, OUTPUT_ROW_BYTES, N_OUT * N_TG,
            )
